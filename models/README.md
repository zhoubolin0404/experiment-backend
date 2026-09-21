# Face occlusion model

`face_attrib_net.onnx` is required by the upload-quality gate in `app.py`.
It is the FaceAttribNet ONNX export distributed by
<https://github.com/yakhyo/face-attribute> and is based on Qualcomm's
Facial-Attribute-Detection model.

The model accepts a centered 128 x 128 RGB face crop with values in `[0, 1]`
and returns five independent probabilities in this order:

1. left eye open
2. right eye open
3. eyeglasses
4. face mask
5. sunglasses

Download location:

<https://github.com/yakhyo/face-attribute/releases/download/weights/face_attrib_net.onnx>

Expected file size: `43,296,036` bytes  
SHA-256: `1BF7C6453BEC2FB28E0830F3A76DCEB9FFD020124F87B28DA5355940A7BC6E48`

The model is distributed under the BSD 3-Clause license included in
`FACE_ATTRIBUTE_MODEL_LICENSE.txt`.
