## Description - export optimized model for RKNPU

### 1. Model structure Adjustment

- The dfl structure has poor performance on NPU processing, moved outside the model.

  Assuming that there are 6000 candidate frames, the original model places the dfl structure before the "box confidence filter", then the 6000 candidate frames need to be calculated through dfl calculation. If the dfl structure is placed after the "box confidence filter", Assuming that there are 100 candidate boxes left after filtering, the calculation amount of the dfl part is reduced to 100, which greatly reduces the occupancy of computing resources and bandwidth resources.

- Notice:  yolov6n/s  hasn't  dfl structure, while yolov6m/l has dfl structure



- Assuming that there are 6000 candidate boxes and the detection category is 80, the threshold retrieval operation needs to be repeated 6000* 80 ~= 4.8*10^5 times, which takes a lot of time. Therefore, when exporting the model, an additional summation operation for 80 types of detection targets is added to the model to quickly filter the confidence. (This structure is effective in some cases, related to the training results of the model)

  (v6m, v6l) To disable this optimization,  comment the following code in ./yolov6/models/effidehead.py (line70~86 part)

  ```
  cls_sum = torch.clamp(y[-1].sum(1, keepdim=True), 0, 1)
  output_for_rknn.append(cls_sum)
  ```

  (v6n, v6s) To disable this optimization,  comment the following code in  ./yolov6/models/heads/effidehead_distill_ns.py (line78~94 part)
  
  ```
  cls_sum = torch.clamp(y[-1].sum(1, keepdim=True), 0, 1)
  output_for_rknn.append(cls_sum)
  ```
  



- This optimization only affects the export of the model and does not affect the training process. **For the training steps, please refer to the YOLOv6 official documentation.**



### 2. Export model operation

After meeting the environmental requirements of ./requirements.txt, execute the following statement to export the model

```
python deploy/RKNN/export_onnx_for_rknn.py --weight ./yolov6n.pt

# adjust ./yolov6n.pt path to export your model.
```



### 3.Transfer to RKNN model, Python demo, C demo

Please refer https://github.com/airockchip/rknn_model_zoo/tree/main/models/CV/object_detection/yolo 