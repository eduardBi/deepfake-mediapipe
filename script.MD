
# python facefusion.py headless-run --target-path  /home/ed/Downloads/2.jpg  --source-paths /home/ed/Downloads/1.jpg  --output-path /home/ed/Downloads/3.jpg

# python facefusion.py headless-run \
#   --target-path /home/ed/Downloads/2.jpg \
#   --source-paths /home/ed/Downloads/1.jpg \
#   --output-path /home/ed/Downloads/3.jpg \
#   --model /home/ed/projects/faceswap/FaceFusion/dataset/inswapper_128.onnx


 
source ~/miniconda3/etc/profile.d/conda.sh
conda activate faceswap


pip install protobuf==3.20.3 --force-reinstall


install jax no avx as well as tensorflow no avx  from https://www.eggwall.com/2022/01/tensorflow-28-and-jax-0176-for-no-avx.html 



 source ~/miniconda3/etc/profile.d/conda.sh
ed@ed:~$ conda create --name deepface python=3.8

