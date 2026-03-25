# Vision Roundtrip Experiments

## Image → Vision Tokens → Image (with WIR modification)

Test whether we can encode an image to a vision-language model's native tokens,
modify those tokens via WIR soft refinement, and decode back to a recognisable image.

### Phase 1: Encode and Invert
- Load a VLM's ViT encoder + mmproj (e.g., Qwen-VL)
- Encode test images to LLM embedding space
- Pseudoinverse the mmproj to recover ViT patch embeddings
- Feature inversion (gradient descent through ViT) to recover rough image
- Measure: is the spatial layout recognisable?

### Phase 2: Diffusion Refinement
- Take blobby inversion output from Phase 1
- Feed as init image to diffusion img2img
- Measure: does diffusion produce a photorealistic version faithful to the original?

### Phase 3: WIR Modification
- Encode a scene to vision tokens
- Run tokens through the VLM with emotional/narrative text context
- WIR soft-refine the vision token embeddings (the model modifies its own visual representation)
- Decode modified embeddings back to image via inversion + diffusion
- Measure: does the output reflect the emotional modification?
  (same scene, but darker/warmer/more threatening based on NPC state)

### Requirements
- A VLM with accessible ViT encoder + mmproj weights (Qwen-VL recommended)
- PyTorch for feature inversion
- A diffusion model for Phase 2+ (Stable Diffusion, Flux, or HY3 VAE)
- Test images (game screenshots or general scenes)
