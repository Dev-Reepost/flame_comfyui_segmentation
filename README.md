# ComfyUI Segmentation Autodesk Flame Pybox Batch Node

An Autodesk Pybox handler for ComfyUI Segment Anything workflow

## Inputs

- `Front` input to ComfyUI EXR loader
  - The image coming from Flame batch upstream node

Input images are written on the ComfyUI server disk
`<COMFYUI_SERVER_MOUNTING>/in/<FLAME_PROJECT>/segmentation/`

## Outputs

- `Result` output from ComfyUI EXR Saver
  - The fully colored-segmentation of the input image
  
- `OutMatte` output from ComfyUI EXR Saver
  - The black & white matte defined by a text prompt

Output images are read on the ComfyUI server disk
`<COMFYUI_SERVER_MOUNTING>/out/<FLAME_PROJECT>/segmentation/<VERSION>/`
