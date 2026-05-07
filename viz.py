import os
import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

from PIL import Image

from network.metrics import mask_dead_pixels

def get_img_from_fig(fig, i,save,dpi=360):
    """
    A function that returns an image as numpy array from a pyplot figure.

    :param fig:
    :param dpi:
    :return:
    """
    buf = io.BytesIO()

    if save: fig.savefig('./results/visualizations2/testing{}.png'.format(i), format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()

    #img = cv2.imdecode(img_arr, 1)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return 0 #img

def get_img_from_fig_event(fig, i,save, root,dpi=360):
    """
    A function that returns an image as numpy array from a pyplot figure.

    :param fig:
    :param dpi:
    :return:
    """
    buf = io.BytesIO()

    name_fig = root + str(i) + ".png"
    if save: fig.savefig(name_fig, format="png", dpi=dpi)

    # if save: fig.savefig('./results/visualizations2/testing{}.png'.format(i), format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()

    #img = cv2.imdecode(img_arr, 1)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return 0 #img


def show_learning(fig,save, chunk, out_depth_potentials, label, title,i):
    """
    On a pyplot figure, confront the outputs of the network with the corresponding groundtruths.

    :param fig:
    :param input_chunk:                         (batchsize, 1, 2, 260, 346)
    :param prediction_out_depth_potentials:     (batchsize, 1, 260, 346)
    :param groundtruth_label:                   (batchsize, 1, 260, 346)
    :param title
    :return:
    """
    
    # 1. Prepare spike histogram for the plot
    frame_ON = chunk[0, :, 0, :].sum(axis=0).cpu().numpy()
    frame_OFF = chunk[0, :, 1, :].sum(axis=0).cpu().numpy()

    frame = np.zeros((260, 346, 3), dtype='int16')

    ON_mask = (frame_ON > 0) & (frame_OFF == 0)
    OFF_mask = (frame_ON == 0) & (frame_OFF > 0)
    ON_OFF_mask = (frame_ON > 0) & (frame_OFF > 0)

    # print("chunk.shape: ", chunk[0, :, 0, :].shape)
    # print("ON_mask.shape: ", ON_mask.shape)
    # print("Frame_ON.shape: ", frame_ON.shape)

    frame[ON_mask] = [255, 0, 0]
    frame[OFF_mask] = [0, 0, 255]
    frame[ON_OFF_mask] = [255, 25, 255]

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4,sharex='col', sharey='row')
    print(title)
    #fig.suptitle(title)
    #fig.axis('off')
    #ax1 = fig.add_subplot(1, 4, 1)

    ax1.set_title('Input spike histogram',fontsize='8')
    #ax1.plot(frame)    
    ax1.imshow(frame)
    # print("unicos numeros input: ", np.unique(frame))
    ax1.axis('off')

    # 2. Prepare network predictions for the plot
    out_depth_potentials, label = mask_dead_pixels(out_depth_potentials, label)

    potentials_copy = out_depth_potentials[-1]
    potentials_copy = potentials_copy.detach().cpu().numpy().squeeze()
    error = np.abs(potentials_copy - label[-1].detach().cpu().numpy().squeeze())
    # error = np.abs(label[-1].detach().cpu().numpy().squeeze() - potentials_copy)

    #ax1 = fig.add_subplot(1, 4, 2)
    ax2.set_title('Prediction',fontsize='8')
    #ax2.plot(potentials_copy)
    ax2.imshow(potentials_copy)
    # print("unicos numeros predicted: ", np.unique(frame))
    ax2.axis('off')

    # 3. Prepare groundtruth map for the plot
    #ax2 = fig.add_subplot(1, 4, 3)
    ax3.set_title('Groundtruth',fontsize='8')
    #ax3.plot(label[-1].detach().cpu().numpy().squeeze())
    ax3.imshow(label[-1].detach().cpu().numpy().squeeze())
    # print("unicos numeros groundtruth: ", np.unique(label[-1].detach().cpu().numpy().squeeze()))
    ax3.axis('off')

    # 4. Also plot the error map (error per pixel)
    #ax3 = fig.add_subplot(1, 4, 4)
    ax4.set_title('Pixel-wise absolute error',fontsize='8')
    #ax4.plot(error)
    ax4.imshow(error)
    ax4.axis('off')
    
    fig.show()
    #plt.draw()

    data = get_img_from_fig(fig,i,save=save, dpi=720)
    # data = get_img_from_fig(fig, dpi=360)

    plt.pause(0.0001)
    plt.clf()

    return 0 #data

def show_learning_eval(fig,save, chunk, out_depth_potentials, title,i):
    """
    On a pyplot figure, confront the outputs of the network with the corresponding groundtruths.

    :param fig:
    :param input_chunk:                         (batchsize, 1, 2, 260, 346)
    :param prediction_out_depth_potentials:     (batchsize, 1, 260, 346)
    :param groundtruth_label:                   (batchsize, 1, 260, 346)
    :param title
    :return:
    """
    
    # 1. Prepare spike histogram for the plot
    frame_ON = chunk[0, :, 0, :].sum(axis=0).cpu().numpy()
    frame_OFF = chunk[0, :, 1, :].sum(axis=0).cpu().numpy()

    frame = np.zeros((260, 346, 3), dtype='int16')

    ON_mask = (frame_ON > 0) & (frame_OFF == 0)
    OFF_mask = (frame_ON == 0) & (frame_OFF > 0)
    ON_OFF_mask = (frame_ON > 0) & (frame_OFF > 0)

    # print("chunk.shape: ", chunk[0, :, 0, :].shape)
    # print("ON_mask.shape: ", ON_mask.shape)
    # print("Frame_ON.shape: ", frame_ON.shape)

    frame[ON_mask] = [255, 0, 0]
    frame[OFF_mask] = [0, 0, 255]
    frame[ON_OFF_mask] = [255, 25, 255]

    fig, (ax1, ax2) = plt.subplots(1, 2,sharex='col', sharey='row')
    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4,sharex='col', sharey='row')
    print(title)
    #fig.suptitle(title)
    #fig.axis('off')
    #ax1 = fig.add_subplot(1, 4, 1)

    ax1.set_title('Input spike histogram',fontsize='8')
    #ax1.plot(frame)    
    ax1.imshow(frame)
    # print("unicos numeros input: ", np.unique(frame))
    ax1.axis('off')

    # 2. Prepare network predictions for the plot
    # out_depth_potentials, label = mask_dead_pixels(out_depth_potentials, label)

    potentials_copy = out_depth_potentials[-1]
    potentials_copy = potentials_copy.detach().cpu().numpy().squeeze()
    # error = np.abs(potentials_copy - label[-1].detach().cpu().numpy().squeeze())
    # # error = np.abs(label[-1].detach().cpu().numpy().squeeze() - potentials_copy)

    #ax1 = fig.add_subplot(1, 4, 2)
    ax2.set_title('Prediction',fontsize='8')
    #ax2.plot(potentials_copy)
    ax2.imshow(potentials_copy)
    # print("unicos numeros predicted: ", np.unique(frame))
    ax2.axis('off')

    # 3. Prepare groundtruth map for the plot
    #ax2 = fig.add_subplot(1, 4, 3)
    # ax3.set_title('Groundtruth',fontsize='8')
    # #ax3.plot(label[-1].detach().cpu().numpy().squeeze())
    # ax3.imshow(label[-1].detach().cpu().numpy().squeeze())
    # # print("unicos numeros groundtruth: ", np.unique(label[-1].detach().cpu().numpy().squeeze()))
    # ax3.axis('off')

    # # 4. Also plot the error map (error per pixel)
    # #ax3 = fig.add_subplot(1, 4, 4)
    # ax4.set_title('Pixel-wise absolute error',fontsize='8')
    # #ax4.plot(error)
    # ax4.imshow(error)
    # ax4.axis('off')
    
    fig.show()
    #plt.draw()

    data = get_img_from_fig(fig,i,save=save, dpi=720)
    # data = get_img_from_fig(fig, dpi=360)

    plt.pause(0.0001)
    plt.clf()

    return 0 #data

def show_learning_pred(fig,save, chunk, out_depth_potentials, label, root, title,i):
    """
    On a pyplot figure, confront the outputs of the network with the corresponding groundtruths.

    :param fig:
    :param input_chunk:                         (batchsize, 1, 2, 260, 346)
    :param prediction_out_depth_potentials:     (batchsize, 1, 260, 346)
    :param groundtruth_label:                   (batchsize, 1, 260, 346)
    :param title
    :return:
    """
    
    # 1. Prepare spike histogram for the plot
    # frame_ON = ch sk] = [255, 25, 255]

    # fig, (ax1, ax2) = plt.subplots(1, 2,sharex='col', sharey='row')
    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4,sharex='col', sharey='row')
    # print(title)
    #fig.suptitle(title)
    #fig.axis('off')
    #ax1 = fig.add_subplot(1, 4, 1)

    # ax1.set_title('Input spike histogram',fontsize='8')
    # #ax1.plot(frame)    
    # ax1.imshow(frame)
    # # print("unicos numeros input: ", np.unique(frame))
    # ax1.axis('off')

    # 2. Prepare network predictions for the plot
    # out_depth_potentials, label = mask_d|ead_pixels(out_depth_potentials, label)

    potentials_copy = out_depth_potentials[-1]
    potentials_copy = potentials_copy.detach().cpu().numpy().squeeze()
    # error = np.abs(potentials_copy - label[-1].detach().cpu().numpy().squeeze())
    # # error = np.abs(label[-1].detach().cpu().numpy().squeeze() - potentials_copy)

    if save:
      img = Image.fromarray(np.uint8(potentials_copy*255))
      img.save(root + str(i) + ".png")

    #ax1 = fig.add_subplot(1, 4, 2)
    # ax2.set_title('Prediction',fontsize='8')
    #ax2.plot(potentials_copy)
    fig = plt.figure()
    plt.imshow(potentials_copy)
    
    plt.show()
    # plt.draw()

    # data = get_img_from_fig_event(fig,i,save=save, root=root, dpi=720)
    # data = get_img_from_fig(fig, dpi=360)

    plt.pause(0.0001)
    plt.clf()

    return 0 #data

def show_loss(fig, input, out, label, title):
    """
    On a pyplot figure, confront the outputs of the network with the corresponding groundtruths.

    :param fig:
    :param input_chunk:                         (batchsize, 1, 2, 260, 346)
    :param prediction_out_depth_potentials:     (batchsize, 1, 260, 346)
    :param groundtruth_label:                   (batchsize, 1, 260, 346)
    :param title
    :return:
    """

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_title('input',fontsize='8')
    ax1.imshow(input[-1].squeeze())

    ax1.axis('off')    

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_title('out',fontsize='8')
    ax2.imshow(out[-1].squeeze())

    ax2.axis('off')

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_title('label',fontsize='8')
    ax3.imshow(label[-1].squeeze())
    ax3.axis('off')

    fig.show()

    plt.pause(0.0001)
    plt.clf()

    return 0 #data    


def make_vid_from_pngs(png_folder, res_tuple, fps, outfile):

    import re

    def atoi(text):
        return int(text) if text.isdigit() else text
        elsetext

    def natural_keys(text):
        return [atoi(c) for c in re.split('(\d+)', text)]

    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    out = cv2.VideoWriter(outfile, fourcc, fps, res_tuple)

    #i = 0
    sorted_filenames = os.listdir(png_folder)
    print("files found: ",len(sorted_filenames))
    #sorted_filenames.sort(key=natural_keys)  # sort png filenames in numerical order
    for File in sorted_filenames:
        #i += 1
        path =png_folder + File
        frame = cv2.imread(path)

        #frame = np.nan_to_num(frame, copy=True,nan=255)  # small processing of the depth maps; DO NOT DO THIS FOR TRAINING A SNN
        #frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
        #frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        #frame =frame.astype(np.uint8)
        #print(path,frame)
        out.write(frame)
        #cv2_imshow(frame)
        cv2.waitKey(int(1000 / fps))

    out.release()
    print("created video file " + outfile)
    print()

# def get_img_from_fig(fig, dpi=180):
#     """
#     A function that returns an image as numpy array from a pyplot figure.

#     :param fig:
#     :param dpi:
#     :return:
#     """
#     buf = io.BytesIO()

#     fig.savefig('./results/visualizations/{}'.format(buf), format="png", dpi=dpi)
#     buf.seek(0)
#     img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
#     buf.close()

#     #img = cv2.imdecode(img_arr, 1)
#     #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     return 0 #img


# def show_learning(fig, chunk, out_depth_potentials, label, title):
#     """
#     On a pyplot figure, confront the outputs of the network with the corresponding groundtruths.

#     :param fig:
#     :param chunk:
#     :param out_depth_potentials: a tensor of shape (batchsize, 1, 260, 346)
#     :param label:  a tensor of shape (batchsize, 1, 260, 346)
#     :param title
#     :return:
#     """
#     plt.title(title)
#     plt.axis('off')
#     #print("ggggggggggggggggggggggggggg")

#     # 1. Prepare spike histogram for the plot
#     frame_ON = chunk[0, :, 0, :].sum(axis=0).cpu().numpy()
#     frame_OFF = chunk[0, :, 1, :].sum(axis=0).cpu().numpy()

#     frame = np.zeros((260, 346, 3), dtype='int16')

#     ON_mask = (frame_ON > 0) & (frame_OFF == 0)
#     OFF_mask = (frame_ON == 0) & (frame_OFF > 0)
#     ON_OFF_mask = (frame_ON > 0) & (frame_OFF > 0)

#     frame[ON_mask] = [255, 0, 0]
#     frame[OFF_mask] = [0, 0, 255]
#     frame[ON_OFF_mask] = [255, 25, 255]

#     ax1 = fig.add_subplot(1, 4, 1)
#     ax1.title.set_text('Input spike histogram')
#     plt.imshow(frame)
#     plt.axis('off')

#     # 2. Prepare network predictions for the plot
#     out_depth_potentials, label = mask_dead_pixels(out_depth_potentials, label)

#     potentials_copy = out_depth_potentials[-1]
#     potentials_copy = potentials_copy.detach().cpu().numpy().squeeze()
#     error = np.abs(potentials_copy - label[-1].detach().cpu().numpy().squeeze())

#     ax1 = fig.add_subplot(1, 4, 2)
#     ax1.title.set_text('Prediction')
#     plt.imshow(potentials_copy)
#     plt.axis('off')

#     # 3. Prepare groundtruth map for the plot
#     ax2 = fig.add_subplot(1, 4, 3)
#     ax2.title.set_text('Groundtruth')
#     plt.imshow(label[-1].detach().cpu().numpy().squeeze())
#     plt.axis('off')

#     # 4. Also plot the error map (error per pixel)
#     ax3 = fig.add_subplot(1, 4, 4)
#     ax3.title.set_text('Pixel-wise absolute error')
#     plt.imshow(error)
#     plt.axis('off')

#     plt.draw()

#     data = get_img_from_fig(fig, dpi=360)

#     plt.pause(0.0001)
#     plt.clf()

#     return 0 #data


# def make_vid_from_pngs(png_folder, res_tuple, fps, outfile):

#     import re

#     def atoi(text):
#         return int(text) if text.isdigit() else text
#         elsetext

#     def natural_keys(text):
#         return [atoi(c) for c in re.split('(\d+)', text)]

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(outfile, fourcc, fps, res_tuple)

#     i = 0
#     sorted_filenames = os.listdir(png_folder)
#     sorted_filenames.sort(key=natural_keys)  # sort png filenames in numerical order
#     for file in sorted_filenames:
#         i += 1
#         frame = cv2.imread(png_folder + file)
#         out.write(frame)
#         cv2.waitKey(int(1000 / fps))

#     out.release()
#     print("created video file " + outfile)
#     print()

