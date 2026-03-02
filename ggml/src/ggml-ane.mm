// ggml-ane.mm — Apple Neural Engine backend implementation
// Uses public CoreML API (MLModel + MLMultiArray) for ANE dispatch.
// Models are FP16 MLProgram compiled via coremltools with opset_version=iOS17.

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <Accelerate/Accelerate.h>
#import <Metal/Metal.h>
#import <mach/mach_time.h>
#import <dispatch/dispatch.h>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <atomic>

#include "ggml-ane.h"

// ===========================================================================
// Timing helper
// ===========================================================================

static double ane_time_ms(void) {
    static mach_timebase_info_data_t tb = {0};
    if (tb.numer == 0) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / tb.denom / 1e6;
}

// ===========================================================================
// ANE Kernel — CoreML model with FP16 MLMultiArray I/O
// ===========================================================================

struct ggml_ane_kernel {
    MLModel * __strong model;
    NSString * __strong inputName;
    NSArray<NSNumber*> * __strong inputShape;
    MLMultiArray * __strong inputArray;
    NSArray<NSString*> * __strong outputNames;
    NSArray<NSArray<NSNumber*>*> * __strong outputShapes;
    id<MLFeatureProvider> __strong lastResult;
    size_t inputBytes;
    bool ownsTemp;
    NSString * __strong tempPath;
};

// ===========================================================================
// ANE availability check
// ===========================================================================

bool ggml_backend_ane_supported(void) {
    // Check for Apple Silicon Neural Engine via CoreML
    if (@available(macOS 14.0, *)) {
        // Try to detect Neural Engine compute device
        Class neClass = NSClassFromString(@"MLNeuralEngineComputeDevice");
        if (!neClass) return false;
        SEL physSel = NSSelectorFromString(@"physicalDevice");
        if (![neClass respondsToSelector:physSel]) return false;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Warc-performSelector-leaks"
        id device = [neClass performSelector:physSel];
#pragma clang diagnostic pop
        return device != nil;
    }
    return false;
}

// ===========================================================================
// Load a pre-compiled .mlmodelc
// ===========================================================================

ggml_ane_kernel_t ggml_ane_load(const char * modelc_path,
                                 const char * input_name,
                                 int compute_units) {
    @autoreleasepool {
        if (!modelc_path) return NULL;

        NSString *path = [NSString stringWithUTF8String:modelc_path];
        NSURL *url = [NSURL fileURLWithPath:path];

        MLModelConfiguration *cfg = [[MLModelConfiguration alloc] init];
        switch (compute_units) {
            case 0: cfg.computeUnits = MLComputeUnitsCPUOnly; break;
            case 1: cfg.computeUnits = MLComputeUnitsCPUAndGPU; break;
            case 2: cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine; break;
            case 3: default: cfg.computeUnits = MLComputeUnitsAll; break;
        }

        NSError *error = nil;
        MLModel *model = [MLModel modelWithContentsOfURL:url
                                           configuration:cfg
                                                   error:&error];
        if (!model) {
            fprintf(stderr, "[ANE] Failed to load %s: %s\n", modelc_path,
                    error ? [[error localizedDescription] UTF8String] : "unknown");
            return NULL;
        }

        // Discover input name and shape
        MLModelDescription *desc = model.modelDescription;
        NSString *inName = input_name ? [NSString stringWithUTF8String:input_name] : nil;
        if (!inName) {
            // Use first input
            for (NSString *name in desc.inputDescriptionsByName) {
                inName = name;
                break;
            }
        }

        MLFeatureDescription *inDesc = desc.inputDescriptionsByName[inName];
        if (!inDesc || !inDesc.multiArrayConstraint) {
            fprintf(stderr, "[ANE] Input '%s' not found or not multiarray\n",
                    inName ? [inName UTF8String] : "nil");
            return NULL;
        }

        NSArray<NSNumber*> *shape = inDesc.multiArrayConstraint.shape;
        MLMultiArrayDataType dtype = inDesc.multiArrayConstraint.dataType;

        // Create reusable input array
        MLMultiArray *inputArr = [[MLMultiArray alloc] initWithShape:shape
                                                           dataType:dtype
                                                              error:&error];
        if (!inputArr) {
            fprintf(stderr, "[ANE] Failed to create input array\n");
            return NULL;
        }

        // Discover output names and shapes
        NSMutableArray<NSString*> *outNames = [NSMutableArray array];
        NSMutableArray<NSArray<NSNumber*>*> *outShapes = [NSMutableArray array];
        for (NSString *name in desc.outputDescriptionsByName) {
            [outNames addObject:name];
            MLFeatureDescription *oDesc = desc.outputDescriptionsByName[name];
            if (oDesc.multiArrayConstraint) {
                [outShapes addObject:oDesc.multiArrayConstraint.shape];
            } else {
                [outShapes addObject:@[]];
            }
        }

        // Calculate input size
        size_t inBytes = 1;
        for (NSNumber *dim in shape) inBytes *= [dim unsignedLongValue];
        inBytes *= (dtype == MLMultiArrayDataTypeFloat16) ? 2 : 4;

        ggml_ane_kernel_t k = (ggml_ane_kernel_t)calloc(1, sizeof(struct ggml_ane_kernel));
        // Retain all ObjC objects — we compile with -fno-objc-arc, so
        // autoreleased objects (e.g. from +modelWithContentsOfURL:) would be
        // freed when the @autoreleasepool exits, leaving dangling pointers.
        k->model        = [model retain];
        k->inputName    = [inName retain];
        k->inputShape   = [shape retain];
        k->inputArray   = [inputArr retain];
        k->outputNames  = [outNames retain];
        k->outputShapes = [outShapes retain];
        k->inputBytes = inBytes;
        k->ownsTemp = false;
        k->tempPath = nil;

        return k;
    }
}

// ===========================================================================
// Compile from weights using Python coremltools
// ===========================================================================

ggml_ane_kernel_t ggml_ane_compile(
    const char * python_path,
    const char * op_type,
    int in_ch, int out_ch, int spatial,
    const float * weight_f32, size_t weight_nbytes,
    const char * cache_dir)
{
    @autoreleasepool {
        if (!op_type) return NULL;

        NSString *pyPath = nil;
        if (python_path) {
            pyPath = [NSString stringWithUTF8String:python_path];
        } else {
            // Auto-detect Python with coremltools installed.
            // Prefer python3.12 (most compatible with coremltools) over newer versions.
            NSArray *candidates = @[
                [NSString stringWithFormat:@"%@/.local/bin/python3.12", NSHomeDirectory()],
                @"/opt/homebrew/bin/python3.12",
                @"/usr/local/bin/python3.12",
                @"/opt/homebrew/bin/python3.11",
                @"/usr/local/bin/python3.11",
                [NSString stringWithFormat:@"%@/.local/bin/python3", NSHomeDirectory()],
                @"/opt/homebrew/bin/python3",
                @"/usr/local/bin/python3",
                @"/usr/bin/python3",
            ];
            // Test with a more thorough check that exercises BlobWriter
            NSString *testScript = @"import coremltools; "
                "from coremltools.converters.mil import Builder as mb; "
                "from coremltools.converters.mil.mil import types; "
                "from coremltools.converters.mil._deployment_compatibility import AvailableTarget as target";
            for (NSString *candidate in candidates) {
                if ([[NSFileManager defaultManager] isExecutableFileAtPath:candidate]) {
                    NSTask *test = [[NSTask alloc] init];
                    test.launchPath = candidate;
                    test.arguments = @[@"-c", testScript];
                    test.standardOutput = [NSPipe pipe];
                    test.standardError = [NSPipe pipe];
                    NSError *testErr = nil;
                    [test launchAndReturnError:&testErr];
                    if (!testErr) {
                        [test waitUntilExit];
                        if (test.terminationStatus == 0) {
                            pyPath = candidate;
                            fprintf(stderr, "[ANE] Using Python: %s\n", [candidate UTF8String]);
                            break;
                        }
                    }
                }
            }
            if (!pyPath) {
                fprintf(stderr, "[ANE] No Python with coremltools found. "
                        "Install: pip3 install coremltools numpy\n");
                return NULL;
            }
        }

        // Create temp/cache directory
        NSString *baseDir;
        if (cache_dir) {
            baseDir = [NSString stringWithUTF8String:cache_dir];
        } else {
            baseDir = [NSTemporaryDirectory() stringByAppendingPathComponent:@"ggml_ane_cache"];
        }
        [[NSFileManager defaultManager] createDirectoryAtPath:baseDir
                                  withIntermediateDirectories:YES
                                                   attributes:nil
                                                        error:nil];

        // Model name based on dimensions
        NSString *modelName = [NSString stringWithFormat:@"%s_%dx%d_sp%d",
                               op_type, in_ch, out_ch, spatial];
        NSString *compiledPath = [baseDir stringByAppendingPathComponent:
            [modelName stringByAppendingString:@".mlmodelc"]];

        // Check if already cached
        if ([[NSFileManager defaultManager] fileExistsAtPath:compiledPath]) {
            ggml_ane_kernel_t k = ggml_ane_load([compiledPath UTF8String], "x", 2);
            if (k) return k;
        }

        // Write weights to temp file
        NSString *weightsPath = [baseDir stringByAppendingPathComponent:
            [modelName stringByAppendingString:@"_weights.bin"]];
        NSData *wData = [NSData dataWithBytes:weight_f32 length:weight_nbytes];
        [wData writeToFile:weightsPath atomically:YES];

        // Write Python generation script
        NSString *scriptPath = [baseDir stringByAppendingPathComponent:@"_gen_model.py"];
        NSString *script = [NSString stringWithFormat:
            @"#!/usr/bin/env python3\n"
            "import sys, os, subprocess, struct, numpy as np\n"
            "import coremltools as ct\n"
            "from coremltools.converters.mil import Builder as mb\n"
            "from coremltools.converters.mil.mil import types\n"
            "from coremltools.converters.mil._deployment_compatibility import AvailableTarget as target\n"
            "\n"
            "op_type = '%s'\n"
            "in_ch, out_ch, spatial = %d, %d, %d\n"
            "weights_path = '%@'\n"
            "out_dir = '%@'\n"
            "model_name = '%@'\n"
            "\n"
            "# Load weights\n"
            "W = np.fromfile(weights_path, dtype=np.float32).reshape(out_ch, in_ch)\n"
            "W16 = W.astype(np.float16).reshape(out_ch, in_ch, 1, 1)\n"
            "\n"
            "if op_type == 'conv':\n"
            "    @mb.program(input_specs=[mb.TensorSpec(shape=(1, in_ch, 1, spatial), dtype=types.fp16)],\n"
            "                opset_version=target.iOS17)\n"
            "    def model(x):\n"
            "        return mb.conv(x=x, weight=mb.const(val=W16, name='W'),\n"
            "                       pad_type='valid', name='out')\n"
            "else:\n"
            "    raise ValueError(f'Unknown op_type: {op_type}')\n"
            "\n"
            "m = ct.convert(model, compute_units=ct.ComputeUnit.ALL,\n"
            "               compute_precision=ct.precision.FLOAT16,\n"
            "               minimum_deployment_target=ct.target.iOS17)\n"
            "pkg = os.path.join(out_dir, model_name + '.mlpackage')\n"
            "m.save(pkg)\n"
            "subprocess.run(['xcrun', 'coremlcompiler', 'compile', pkg, out_dir], check=True)\n"
            "print('OK')\n",
            op_type, in_ch, out_ch, spatial,
            weightsPath, baseDir, modelName];
        [script writeToFile:scriptPath atomically:YES encoding:NSUTF8StringEncoding error:nil];

        // Run Python
        NSTask *task = [[NSTask alloc] init];
        task.launchPath = [pyPath UTF8String] ? pyPath : @"/usr/bin/python3";
        task.arguments = @[scriptPath];
        // Redirect stdout to /dev/null (coremltools is verbose).
        // Capture stderr via pipe but read it BEFORE waitUntilExit to avoid
        // pipe-buffer deadlocks (Apple docs: "read the pipe before waiting").
        task.standardOutput = [NSFileHandle fileHandleWithNullDevice];
        NSPipe *errPipe = [NSPipe pipe];
        task.standardError = errPipe;

        NSError *error = nil;
        [task launchAndReturnError:&error];
        if (error) {
            fprintf(stderr, "[ANE] Failed to launch Python: %s\n",
                    [[error localizedDescription] UTF8String]);
            return NULL;
        }

        // Read stderr fully before waitUntilExit to prevent pipe-buffer deadlock.
        NSData *errData = [[errPipe fileHandleForReading] readDataToEndOfFile];
        [task waitUntilExit];

        if (task.terminationStatus != 0) {
            NSString *errStr = [[NSString alloc] initWithData:errData encoding:NSUTF8StringEncoding];
            fprintf(stderr, "[ANE] Python model gen failed: %s\n", [errStr UTF8String]);
            return NULL;
        }

        // Clean up temp files
        [[NSFileManager defaultManager] removeItemAtPath:weightsPath error:nil];
        [[NSFileManager defaultManager] removeItemAtPath:scriptPath error:nil];

        // Load the compiled model
        ggml_ane_kernel_t k = ggml_ane_load([compiledPath UTF8String], "x", 2);
        if (k) {
            k->ownsTemp = (cache_dir == NULL);
            k->tempPath = [baseDir retain];
        }
        return k;
    }
}

// ===========================================================================
// I/O operations
// ===========================================================================

void ggml_ane_write_input(ggml_ane_kernel_t k, int idx, const void * data, size_t nbytes) {
    if (!k || idx != 0 || !data) return;
    // Direct memcpy into MLMultiArray's data pointer
    void *dst = k->inputArray.dataPointer;
    size_t copyBytes = (nbytes < k->inputBytes) ? nbytes : k->inputBytes;
    memcpy(dst, data, copyBytes);
}

void ggml_ane_read_output(ggml_ane_kernel_t k, int idx, void * data, size_t nbytes) {
    // Output reading is done after eval via ggml_ane_eval
    // This is a placeholder — actual output reading happens in eval
    (void)k; (void)idx; (void)data; (void)nbytes;
}

// ===========================================================================
// Eval — run prediction on ANE
// ===========================================================================

bool ggml_ane_eval(ggml_ane_kernel_t k) {
    if (!k || !k->model) return false;
    @autoreleasepool {
        NSError *error = nil;
        id prov = [[MLDictionaryFeatureProvider alloc]
            initWithDictionary:@{k->inputName: k->inputArray}
            error:&error];
        if (!prov) {
            fprintf(stderr, "[ANE] Failed to create feature provider: %s\n",
                    error ? [[error localizedDescription] UTF8String] : "unknown");
            return false;
        }

        id<MLFeatureProvider> result = [k->model predictionFromFeatures:prov error:&error];
        [prov release];
        if (!result) {
            fprintf(stderr, "[ANE] Prediction failed: %s\n",
                    error ? [[error localizedDescription] UTF8String] : "unknown");
            return false;
        }

        [k->lastResult release];
        k->lastResult = [result retain];
        return true;
    }
}

// Read output after eval. idx selects the output (0-based, in order of outputNames).
// Copies FP16 data into caller's buffer.
void ggml_ane_read_output_after_eval(ggml_ane_kernel_t k, int idx,
                                      void * data, size_t nbytes) {
    if (!k || !k->lastResult || idx < 0 || idx >= (int)[k->outputNames count]) return;
    NSString *outName = k->outputNames[idx];
    MLFeatureValue *fv = [k->lastResult featureValueForName:outName];
    if (!fv || !fv.multiArrayValue) return;

    MLMultiArray *arr = fv.multiArrayValue;
    size_t available = arr.count * 2; // FP16 = 2 bytes per element
    size_t copyBytes = (nbytes < available) ? nbytes : available;
    memcpy(data, arr.dataPointer, copyBytes);
}

// ===========================================================================
// Utility functions
// ===========================================================================

int ggml_ane_n_outputs(ggml_ane_kernel_t k) {
    if (!k) return 0;
    return (int)[k->outputNames count];
}

const char * ggml_ane_output_name(ggml_ane_kernel_t k, int idx) {
    if (!k || idx < 0 || idx >= (int)[k->outputNames count]) return NULL;
    return [k->outputNames[idx] UTF8String];
}

void ggml_ane_free(ggml_ane_kernel_t k) {
    if (!k) return;
    @autoreleasepool {
        // Release retained ObjC objects (we use -fno-objc-arc, so manual retain/release).
        [k->model release];
        [k->inputArray release];
        [k->inputName release];
        [k->inputShape release];
        [k->outputNames release];
        [k->outputShapes release];
        [k->lastResult release];
        k->model = nil;
        k->inputArray = nil;
        k->inputName = nil;
        k->inputShape = nil;
        k->outputNames = nil;
        k->outputShapes = nil;
        k->lastResult = nil;
        if (k->ownsTemp && k->tempPath) {
            [[NSFileManager defaultManager] removeItemAtPath:k->tempPath error:nil];
        }
        [k->tempPath release];
        k->tempPath = nil;
        free(k);
    }
}

// ===========================================================================
// Weight blob builders (compatible with coremltools format)
// ===========================================================================

void * ggml_ane_build_weight_blob(const float * weights_f32,
                                   int out_ch, int in_ch,
                                   size_t * out_nbytes) {
    size_t wsize = (size_t)out_ch * in_ch * 2; // FP16
    size_t total = 64 + 64 + wsize; // global header + chunk header + data
    uint8_t *buf = (uint8_t*)calloc(total, 1);

    // Global header
    buf[0] = 0x01; buf[4] = 0x02;
    // Chunk header at offset 64
    uint8_t *chunk = buf + 64;
    chunk[0] = 0xEF; chunk[1] = 0xBE; chunk[2] = 0xAD; chunk[3] = 0xDE;
    chunk[4] = 0x01;
    *(uint32_t*)(chunk + 8) = (uint32_t)wsize;
    *(uint32_t*)(chunk + 16) = 128; // data_offset from file start

    // Convert F32 → FP16
    _Float16 *fp16 = (_Float16*)(buf + 128);
    for (size_t i = 0; i < (size_t)out_ch * in_ch; i++) {
        fp16[i] = (_Float16)weights_f32[i];
    }

    *out_nbytes = total;
    return buf;
}

void ggml_ane_free_string(char * s) {
    free(s);
}

// ===========================================================================
// Benchmarking / characterization
// ===========================================================================

float ggml_ane_measure_tflops(ggml_ane_kernel_t k,
                               int channels, int spatial,
                               int n_iters) {
    if (!k) return 0.0f;
    @autoreleasepool {
        // Fill input with small values
        __fp16 *ip = (__fp16*)k->inputArray.dataPointer;
        for (size_t i = 0; i < k->inputArray.count; i++) ip[i] = (__fp16)0.01f;

        // Warmup
        for (int i = 0; i < 10; i++) ggml_ane_eval(k);

        // Benchmark
        double t0 = ane_time_ms();
        for (int i = 0; i < n_iters; i++) ggml_ane_eval(k);
        double elapsed_ms = ane_time_ms() - t0;

        double flops_per_eval = 2.0 * channels * channels * spatial;
        double total_flops = flops_per_eval * n_iters;
        return (float)(total_flops / (elapsed_ms / 1000.0) / 1e12);
    }
}

float ggml_ane_measure_dispatch_latency(ggml_ane_kernel_t k, int n_iters) {
    if (!k) return -1.0f;
    @autoreleasepool {
        __fp16 *ip = (__fp16*)k->inputArray.dataPointer;
        for (size_t i = 0; i < k->inputArray.count; i++) ip[i] = (__fp16)0.01f;

        // Warmup
        for (int i = 0; i < 20; i++) ggml_ane_eval(k);

        double t0 = ane_time_ms();
        for (int i = 0; i < n_iters; i++) ggml_ane_eval(k);
        double elapsed_ms = ane_time_ms() - t0;

        return (float)(elapsed_ms / n_iters);
    }
}

float ggml_ane_test_concurrent_metal(ggml_ane_kernel_t ak) {
    if (!ak) return -1.0f;
    @autoreleasepool {
        __fp16 *ip = (__fp16*)ak->inputArray.dataPointer;
        for (size_t i = 0; i < ak->inputArray.count; i++) ip[i] = (__fp16)0.01f;

        // Setup Metal busy-work shader
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return -1.0f;

        NSString *shaderSrc = @
            "#include <metal_stdlib>\n"
            "using namespace metal;\n"
            "kernel void busy(device float *buf [[buffer(0)]], uint id [[thread_position_in_grid]]) {\n"
            "    float x = buf[id];\n"
            "    for (int i = 0; i < 100000; i++) { x = fma(x, 1.00001f, 0.00001f); }\n"
            "    buf[id] = x;\n"
            "}\n";

        NSError *err = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:shaderSrc options:nil error:&err];
        if (!lib) return -1.0f;
        id<MTLFunction> func = [lib newFunctionWithName:@"busy"];
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:func error:&err];

        int metal_n = 65536;
        id<MTLBuffer> mbuf = [device newBufferWithLength:metal_n * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
        float *mptr = (float*)[mbuf contents];
        for (int i = 0; i < metal_n; i++) mptr[i] = 1.0f;

        id<MTLCommandQueue> queue = [device newCommandQueue];

        // Warmup
        for (int i = 0; i < 3; i++) ggml_ane_eval(ak);
        {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pipeline];
            [enc setBuffer:mbuf offset:0 atIndex:0];
            [enc dispatchThreads:MTLSizeMake(metal_n, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        }

        int n_trials = 5;

        // Metal-only
        double metal_ms = 0;
        for (int t = 0; t < n_trials; t++) {
            double t0 = ane_time_ms();
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pipeline];
            [enc setBuffer:mbuf offset:0 atIndex:0];
            [enc dispatchThreads:MTLSizeMake(metal_n, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
            metal_ms += ane_time_ms() - t0;
        }
        metal_ms /= n_trials;

        // ANE-only
        double ane_ms = 0;
        for (int t = 0; t < n_trials; t++) {
            double t0 = ane_time_ms();
            ggml_ane_eval(ak);
            ane_ms += ane_time_ms() - t0;
        }
        ane_ms /= n_trials;

        // Concurrent
        double concurrent_ms = 0;
        for (int t = 0; t < n_trials; t++) {
            dispatch_group_t group = dispatch_group_create();
            double t0 = ane_time_ms();

            dispatch_group_enter(group);
            dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
                ggml_ane_eval(ak);
                dispatch_group_leave(group);
            });

            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pipeline];
            [enc setBuffer:mbuf offset:0 atIndex:0];
            [enc dispatchThreads:MTLSizeMake(metal_n, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];

            dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
            concurrent_ms += ane_time_ms() - t0;
        }
        concurrent_ms /= n_trials;

        double max_time = fmax(metal_ms, ane_ms);
        float ratio = (float)(concurrent_ms / max_time);

        fprintf(stderr, "[ANE] Concurrency: metal=%.2fms ane=%.2fms concurrent=%.2fms ratio=%.3f (%s)\n",
                metal_ms, ane_ms, concurrent_ms, ratio,
                ratio < 1.1 ? "CONCURRENT" : "SERIALIZED");

        return ratio;
    }
}

// ===========================================================================
// Backend stubs — ANE runs as co-processor side-channel, not full ggml backend
// ===========================================================================

ggml_backend_t ggml_backend_ane_init(void) {
    return NULL;
}

bool ggml_backend_is_ane(ggml_backend_t backend) {
    (void)backend;
    return false;
}

ggml_backend_buffer_type_t ggml_backend_ane_buffer_type(void) {
    return NULL;
}
