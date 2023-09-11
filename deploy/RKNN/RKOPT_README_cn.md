## RKNN 导出模型说明

### 1.调整部分

- 由于 dfl 结构在 npu 处理性能不佳。假设有6000个候选框，原模型将 dfl 结构放置于 ''框置信度过滤" 前，则 6000 个候选框都需要计算经过 dfl 计算；而将 dfl 结构放置于 ''框置信度过滤" 后，假设过程成 100 个候选框，则dfl部分计算量减少至 100 个。

  故将 dfl 结构使用 cpu 处理的耗时，虽然享受不到 npu 加速，但是本来带来的计算量较少也是很可观的。
  
  注： yolov6n, yolov6s 没有 dfl 结构; yolov6m, yolov6l 存在 dfl 结构



- 假设存在 6000 个候选框，存在 80 类检测目标，则阈值需要检索的置信度有 6000* 80 ～= 4.8*10^5 个，占据了较多耗时，故导出模型时，在模型中额外新增了对 80 类检测目标进行求和操作，用于快速过滤置信度，该结构在部分情况下对模型有效。

  (v6m, v6l) 可以在 ./yolov6/models/effidehead.py 70~86行位置，注释掉这部分

  ```
  cls_sum = torch.clamp(y[-1].sum(1, keepdim=True), 0, 1)
  output_for_rknn.append(cls_sum)
  ```

  (v6n, v6s) 可以在  yolov6/models/heads/effidehead_distill_ns.py 78~94行位置，注释掉这部分
  
  ```
  cls_sum = torch.clamp(y[-1].sum(1, keepdim=True), 0, 1)
  output_for_rknn.append(cls_sum)
  ```
  



- 以上优化只影响了模型的导出，不影响训练过程，**训练步骤请参考 YOLOv6 官方文档**。




### 2.导出模型操作

在满足 ./requirements.txt 的环境要求后，执行以下语句导出模型

```
python deploy/RKNN/export_onnx_for_rknn.py --weight ./yolov6n.pt

# 如果自己训练模型，则路径./yolov6n.pt 请改为自己模型的路径
```



### 3.转RKNN模型、Python demo、C demo

请参考 https://github.com/airockchip/rknn_model_zoo/tree/main/models/CV/object_detection/yolo 