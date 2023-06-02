## 复现Adabins并完成onnx导出与ONNXruntime的推理：
* github地址：https://github.com/shariqfarooq123/AdaBins
* 论文地址：https://arxiv.org/pdf/2011.14141.pdf
### 1. 环境配置及pytorch上复现：
* 系统环境：
  * ubuntu18.04
  * CUDA-11.1
  * FFMPEG  3.4.11 (装了就行，没必要我这个版本)
* 项目的依赖应该不是很复杂，这边就列举我用的conda虚拟环境的几个关键的包及其版本：
  ```
  onnx                          1.13.0
  onnx-simplifier               0.4.10
  onnxruntime                   1.12.1
  onnxruntime-gpu               1.14.1
  numpy                         1.23.2
  matplotlib                    3.6.0
  torch                         1.12.1+cu113
  torchvision                   0.13.1+cu113
  tqdm                          4.64.1
  Pillow                        9.2.0
  ```
* pytorch端复现：
  * 1.克隆项目
    ```shell
    git clone https://github.com/shariqfarooq123/AdaBins.git
    cd AdaBins
    ```
  * 2.下载权重：
    * 权重地址：[地址](https://drive.google.com/drive/folders/1nYyaQXOBjNdUJDsmJpcRpu6oE55aQoLA?usp=sharing)
    * 需要魔法，本人直接打包在百度网盘，权重较大；
      ```shell
      mkdir pretrained
      ```
    * 将刚才下载好的权重`AdaBins_nyu.pt`放到`pretrained`目录下
  * 3.处理原视频`IMG_1662.MOV`
    * 原视频是4K的分辨率，较大，我们需要把视频的分辨率稍微降低一些，否则会爆显存；
    * 这里我们采用`FFPMEG`将视频的分辨率调低：
      ```shell
      ffmpeg -i IMG_1662.MOV -vf scale=960:540 video_540.mp4
      ```
    * 生成`video_540.mp4`，此时将视频保存为图片，先将流程走通；依旧使用 FFMPEG：
      ```shell
      mkdir img
      ffmpeg -i video_540.mp4 img/%4d.jpg
      ```
      * 则会在img文件夹下面生成290张图片
  * 4.运行推理脚本
    * 运行脚本：
      ```shell
      python infer.py
      ```
    * 即可在目录`video/out`下生成逐帧的预测图片
  * 5.图片转视频:注意里面的路径
    ```
    python image2video.py
    ```
  * 此时即可完成pytorch端的推理并转换成视频

### 导出ONNX以及ONNXRuntime的推理：
* 1.将模型导出为ONNX并简化ONNX：
  ```shell
  python export.py
  ```
  * 此时生成两个模型，一个`model.onnx`,一个是`sim.onnx`，我们使用后者在onnxruntime中进行推理
* 2.利用onnxruntime进行推理：
  ```shell
  python ort_infer.py
  ```
  * 执行该脚本即可完成类似pytorch中的操作，在某个文件夹下面生成逐帧推理的结果
* 3.图片转视频:注意里面的路径
    ```
    python image2video.py
    ```
