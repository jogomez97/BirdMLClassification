import PySimpleGUI as sg
from PIL import Image
import os.path
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from cv2 import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pylab as plt2
from tensorflow.keras.models import load_model as tflm

def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg

# sg.theme('BlueMono')
sg.theme('Reddit')
model_labels_column = [
    [sg.Text("Model file"), sg.In(size=(25, 1), enable_events=True, key="-MODEL FILE-"), sg.FilesBrowse(file_types=(("Model files (.h5)", "*.h5"),))],
    [sg.Text("Labels File"), sg.In(size=(25, 1), enable_events=True, key="-LABELS FILE-"), sg.FilesBrowse(file_types=(("Text Files", "*.txt"),))],
    [sg.Listbox(values=[], enable_events=False, size=(40, 20), key="-LABELS LIST-")],
]

file_list_column = [
    [sg.Text("Image Folder"), sg.In(size=(25, 1), enable_events=True, key="-IMG FOLDER-"), sg.FolderBrowse(),],
    [sg.Listbox(values=[], enable_events=True, size=(40, 20), key="-FILE LIST-")],
]

image_viewer_column = [
    [sg.Text("Choose an image from list on left:")],
    [sg.Text(size=(60, 1), key="-TOUT-")],
    [sg.Button(button_text="Predict", key="-PREDICT-")],
    [sg.Text("Prediction: ",size=(40, 1), key="-PRED-")],
    [sg.Canvas(size=(480, 640), key="-CANVAS-")],
]


# ----- Full layout -----
layout = [[
        sg.Column(model_labels_column),
        sg.VSeperator(),
        sg.Column(file_list_column),
        sg.VSeperator(),
        sg.Column(image_viewer_column),
    ]]


window = sg.Window("Machine learning predictor", layout, finalize=True)


has_model = False 
has_image = False 
has_labels = False

# Draw initial plot in the window
canvas_elem = window['-CANVAS-']
canvas = canvas_elem.TKCanvas
fig, ax = plt.subplots(1, 1)
ax.set_ylabel("%"); ax.set_ylim(bottom=0, top=100)
fig_pred = draw_figure(canvas, fig)

# Run the Event Loop
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break

    if event == "-IMG FOLDER-":
        folder = values["-IMG FOLDER-"]
        try:
            file_list = os.listdir(folder)
        except:
            file_list = []
        fnames = [f for f in file_list if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith((".png", ".jpg", ".jpeg", ".npy"))]
        window["-FILE LIST-"].update(fnames)

    elif event == "-MODEL FILE-":
        model_file = values["-MODEL FILE-"]
        if model_file.lower().endswith(".h5"):
            try:
                model = load_model(model_file)
                has_model = True
            except:
                try:
                    model = tflm(model_file)
                    has_model = True
                except Exception as e:
                    has_model = False
                    sg.Popup("Error loading the model. Error: {}".format(e))


    elif event == "-LABELS FILE-":
        if values["-LABELS FILE-"] == "":
            continue
        try:
            f = open(values["-LABELS FILE-"] , "r")
            aux = f.readlines()
            labels = []
            for l in aux:
                labels.append(l.replace('\n', ''))
            f.close()
            window["-LABELS LIST-"].update(labels)
            has_labels = True
        except Exception as e:
            has_labels = False
            sg.Popup("Error loading labels.  Error: {}".format(e))

    elif event == "-FILE LIST-":  
        try:
            window["-PRED-"].update("Prediction: ")
            window["-TOUT-"].update(values["-FILE LIST-"][0])

            ax.cla()
            ax.set_ylabel("%"); ax.set_ylim(bottom=0, top=100)
            fig_pred.draw()

            if values["-FILE LIST-"][0] != "":
                has_image = True
        except Exception as e:
            print(e)
            pass
    
    elif event == "-PREDICT-":
        # print(has_model, has_labels, has_image)
        if has_model and has_labels and has_image:
            filename = os.path.join(values["-IMG FOLDER-"], values["-FILE LIST-"][0])
            if filename.lower().endswith(".npy"):
                image = np.load(filename) / 255.0
                image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
                image = np.expand_dims(image, axis=0)
            else:
                image = Image.open(filename)
                image = image.resize((224, 224), Image.LANCZOS)
                image = image.convert('RGB')
                # image.save('out_img.png')
                image = np.array(image)
                image = np.expand_dims(image, axis=0)
            
            preds = model.predict(image)[0]
            i = np.argmax(preds)
            label = labels[i]
            preds = preds * 100
            
            ax.cla()            
            ax.bar(labels,preds,align="center",alpha=.5)
            ax.set_xticks(labels)
            ax.set_ylabel("%"); ax.set_ylim(bottom=0, top=100)
            plt.setp(ax.get_xticklabels(), rotation=60, horizontalalignment='right')
            plt.tight_layout()
            fig_pred.draw()

            msg = "Prediction: {} ({:.2f}%)".format(label, preds[i])
            window["-PRED-"].update(msg)
        else:
            sg.Popup("Please set model, labels, images and select one image")

window.close()