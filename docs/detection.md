# Object Detection and Semantic Segmentation

Due to [the changes](../mmdetection/docs/compatibility.md) between MMDetection v1.x and v2.x, models directly converted from our previous implementation at [AttentiveNorm_Detection](https://github.com/iVMCL/AttentiveNorm_Detection) suffer significant performance drop.

We will train models using this refactored code and release the trained models here.

- [ ] **All the models are released on Google Drive** ([download link](https://drive.google.com/drive/folders/1kirh0AzZedaxUVy6AsheUeIsuH7U8TXH?usp=sharing))

## Mask-RCNN
<table>
  <tr>
    <th>Architecture</th>
    <th>Backbone</th>
    <th>Head</th>
    <th>box AP</th>
    <th>mask AP</th>
    <th>Remarks</th>
  </tr>
  <tr>
    <td rowspan="5">ResNet-50</td>
    <td>BN</td>
    <td>-</td>
    <td>39.2</td>
    <td>35.4</td>
    <td><a href="https://github.com/open-mmlab/mmdetection/tree/master/configs/mask_rcnn">MMDet trained</a></td>
  </tr>
  <tr>
    <td>AN (w/ BN)</td>
    <td>-</td>
    <td>40.8</td>
    <td>36.4</td>
    <td><a href="https://github.com/iVMCL/AttentiveNorm_Detection">AttentiveNorm_Detection trained</a></td>
  </tr>
  <tr>
    <td>AN (w/ BN)</td>
    <td>-</td>
    <td>41.5</td>
    <td>36.8</td>
    <td>This repo</td>
  </tr>
  <tr>
    <td>*GN</td>
    <td>GN</td>
    <td>40.3</td>
    <td>35.7</td>
  </tr>
    <tr>
    <td>*SN</td>
    <td>SN</td>
    <td>41.0</td>
    <td>36.5</td>
</table>
