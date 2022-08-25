#화질 개선- 이미지가 있는 경로를 활용해서 결과 도출
#1) Yolo crop 이미지를 불러와서 화질 개선 가능
#2) 구글 드라이브에 있는 이미지를 불러와서 화질 개선 가능
#yolo 결과물로 만들때는 img_route 끝에 '/' 추가 필요!

def view_clear(main_route,img_route):
    import os
    os.chdir(main_route)
    # Clone realESRGAN
    !git clone https://github.com/xinntao/Real-ESRGAN.git
    %cd Real-ESRGAN
    # Set up the environment
    !pip install basicsr
    !pip install facexlib
    !pip install gfpgan
    !pip install -r requirements.txt
    !python setup.py develop

    # Clone BSRGAN
    !git clone https://github.com/cszn/BSRGAN.git

    !rm -r SwinIR
    # Clone SwinIR
    !git clone https://github.com/JingyunLiang/SwinIR.git
    !pip install timm

    #download the pre-trained model
    !wget https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth -P experiments/pretrained_models

    #upload image
    import os
    from google.colab import files
    import shutil
    from glob import glob

    # test SwinIR by partioning the image into patches
    test_patch_wise = False

    #이미 파일이 있으면-> 폴더 비우기
    # to be compatible with BSRGAN
    !rm -r BSRGAN/testsets/RealSRSet
    upload_folder = 'BSRGAN/testsets/RealSRSet'
    result_folder = 'results'

    #지정 경로에 이미지 있으면 확인하고 지우기
    if os.path.isdir(upload_folder):
        shutil.rmtree(upload_folder)
    if os.path.isdir(result_folder):
        shutil.rmtree(result_folder)
    os.mkdir(upload_folder)
    os.mkdir(result_folder)

    #경로 이동해서 분석하는 파일로 옮기기
    num=len(glob(img_route+'/*.jpg'))
    shutil.copy(glob(img_route+'/*.jpg')[int(num/2)], 'BSRGAN/testsets/RealSRSet')

    # SwinIR-Large
    if test_patch_wise:
      !python SwinIR/main_test_swinir.py --task real_sr --model_path experiments/pretrained_models/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth --folder_lq BSRGAN/testsets/RealSRSet --scale 4 --large_model --tile 640
    else:
      !python SwinIR/main_test_swinir.py --task real_sr --model_path experiments/pretrained_models/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth --folder_lq BSRGAN/testsets/RealSRSet --scale 4 --large_model
    shutil.move('results/swinir_real_sr_x4_large', 'results/SwinIR_large')
    for path in sorted(glob(os.path.join('results/SwinIR_large', '*.png'))):
      os.rename(path, path.replace('SwinIR.png', 'SwinIR_large.png'))

    #Visualization
    import cv2
    from google.colab.patches import cv2_imshow
    im0=cv2.imread(glob(main_route+'/Real-ESRGAN/results/SwinIR_large/*.png')[0])
    cv2_imshow(im0)

    #Move result to Drive
    shutil.copy(glob(main_route+'/Real-ESRGAN/results/SwinIR_large/*.png')[0], img_route)