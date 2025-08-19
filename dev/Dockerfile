FROM nvidia/cuda:12.6.3-py3.12.1 

# Set the working directory 
WORKDIR /app 


RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu126 
RUN pip3 install --no-cache-dir vllm==0.10.0 
RUN pip3 install --no-cache-dir setuptools==79.0.1 
Run pip3 install --no-cache-dir opencv-python-headless==4.11.0.86 latex2sympy2_extended==1.0.6 
RUN pip3 install --no-cache-dir flash-attn --no-build-isolation 
RUN pip3 install --no-cache-dir lighteval==0.9.2 
RUN pip3 install --no-cache-dir math-verify==0.5.2 
RUN pip3 install --no-cache-dir fsspec==2025.3.0 s3fs==2025.3.0 datasets==3.6.0 
Run pip3 install --no-cache-dir more-itertools 

COPY . /app