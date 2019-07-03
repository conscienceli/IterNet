import numpy as np


def get_test_patches(img, crop_size, stride_size,rl=False):
    """
    将待分割图预处理后，分割成patch
    :param img: 待分割图
    :return:
    """
    test_img = []

    test_img.append(img)
    test_img=np.asarray(test_img)

#     test_img_adjust=img_process(test_img,rl=rl)  #预处理
    test_img_adjust = test_img
    test_imgs=paint_border(test_img_adjust,crop_size, stride_size)  #将图片补足到可被完美分割状态

    test_img_patch=extract_patches(test_imgs,crop_size, stride_size)  #依顺序分割patch

    return test_img_patch,test_imgs.shape[1],test_imgs.shape[2],test_img_adjust


def extract_patches(full_imgs, crop_size, stride_size):
    """
    按顺序分割patch
    :param full_imgs: 补足后的图片
    :return: 分割后的patch
    """
    patch_height = crop_size
    patch_width = crop_size
    stride_height = stride_size
    stride_width = stride_size
    
    
    assert (len(full_imgs.shape)==4)  #4D arrays
    img_h = full_imgs.shape[1]  #height of the full image
    img_w = full_imgs.shape[2] #width of the full image

    assert ((img_h-patch_height)%stride_height==0 and (img_w-patch_width)%stride_width==0)
    N_patches_img = ((img_h-patch_height)//stride_height+1)*((img_w-patch_width)//stride_width+1)  #// --> division between integers
    N_patches_tot = N_patches_img*full_imgs.shape[0]

    patches = np.empty((N_patches_tot,patch_height,patch_width,full_imgs.shape[3]))
    iter_tot = 0   #iter over the total number of patches (N_patches)
    for i in range(full_imgs.shape[0]):  #loop over the full images
        for h in range((img_h-patch_height)//stride_height+1):
            for w in range((img_w-patch_width)//stride_width+1):
                patch = full_imgs[i,h*stride_height:(h*stride_height)+patch_height,w*stride_width:(w*stride_width)+patch_width,:]
                patches[iter_tot]=patch
                iter_tot +=1   #total
    assert (iter_tot==N_patches_tot)
    return patches


def paint_border(imgs,crop_size, stride_size):
    """
    将图片补足到可被完美分割状态
    :param imgs:  预处理后的图片
    :return:
    """
    patch_height = crop_size
    patch_width = crop_size
    stride_height = stride_size
    stride_width = stride_size
    
    assert (len(imgs.shape) == 4)
    img_h = imgs.shape[1]  # height of the full image
    img_w = imgs.shape[2]  # width of the full image
    leftover_h = (img_h - patch_height) % stride_height  # leftover on the h dim
    leftover_w = (img_w - patch_width) % stride_width  # leftover on the w dim
    full_imgs=None
    if (leftover_h != 0):  #change dimension of img_h
        tmp_imgs = np.zeros((imgs.shape[0],img_h+(stride_height-leftover_h),img_w,imgs.shape[3]))
        tmp_imgs[0:imgs.shape[0],0:img_h,0:img_w,0:imgs.shape[3]] = imgs
        full_imgs = tmp_imgs
    if (leftover_w != 0):   #change dimension of img_w
        tmp_imgs = np.zeros((full_imgs.shape[0],full_imgs.shape[1],img_w+(stride_width - leftover_w),full_imgs.shape[3]))
        tmp_imgs[0:imgs.shape[0],0:imgs.shape[1],0:img_w,0:full_imgs.shape[3]] =imgs
        full_imgs = tmp_imgs
#     print("new full images shape: \n" +str(full_imgs.shape))
        return full_imgs
    else:
        return imgs




def pred_to_patches(pred, crop_size, stride_size):
    """
    将预测的向量 转换成patch形态
    :param pred: 预测结果
    :param config: 配置文件
    :return: Tensor [-1，patch_height,patch_width,seg_num+1]
    """
    return pred
    patch_height = crop_size
    patch_width = crop_size
    
    seg_num = 0
#     print(pred.shape)
    
    assert (len(pred.shape)==3)  #3D array: (Npatches,height*width,2)

    pred_images = np.empty((pred.shape[0],pred.shape[1],seg_num+1))  #(Npatches,height*width)
    pred_images[:,:,0:seg_num+1]=pred[:,:,0:seg_num+1]
    pred_images = np.reshape(pred_images,(pred_images.shape[0],patch_height,patch_width,seg_num+1))
    return pred_images




def recompone_overlap(preds,crop_size,stride_size,img_h,img_w):
    """
    将patch拼成原始图片
    :param preds: patch块
    :param img_h:  原始图片 height
    :param img_w:  原始图片 width
    :return:  拼接成的图片
    """
    assert (len(preds.shape)==4)  #4D arrays

    patch_h = crop_size
    patch_w = crop_size
    stride_height = stride_size
    stride_width = stride_size
    
    N_patches_h = (img_h-patch_h)//stride_height+1
    N_patches_w = (img_w-patch_w)//stride_width+1
    N_patches_img = N_patches_h * N_patches_w
#     print("N_patches_h: " +str(N_patches_h))
#     print("N_patches_w: " +str(N_patches_w))
#     print("N_patches_img: " +str(N_patches_img))
    #assert (preds.shape[0]%N_patches_img==0)
    N_full_imgs = preds.shape[0]//N_patches_img
#     print("According to the dimension inserted, there are " +str(N_full_imgs) +" full images (of " +str(img_h)+"x" +str(img_w) +" each)")
    full_prob = np.zeros((N_full_imgs,img_h,img_w,preds.shape[3]))  #itialize to zero mega array with sum of Probabilities
    full_sum = np.zeros((N_full_imgs,img_h,img_w,preds.shape[3]))

    k = 0 #iterator over all the patches
    for i in range(N_full_imgs):
        for h in range((img_h-patch_h)//stride_height+1):
            for w in range((img_w-patch_w)//stride_width+1):
                full_prob[i,h*stride_height:(h*stride_height)+patch_h,w*stride_width:(w*stride_width)+patch_w,:]+=preds[k]
                full_sum[i,h*stride_height:(h*stride_height)+patch_h,w*stride_width:(w*stride_width)+patch_w,:]+=1
                k+=1
#     print(k,preds.shape[0])
    assert(k==preds.shape[0])
    assert(np.min(full_sum)>=1.0)  #at least one
    final_avg = full_prob/full_sum
#     print('using avg')
    return final_avg