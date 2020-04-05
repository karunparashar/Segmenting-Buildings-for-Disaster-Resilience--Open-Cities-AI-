from tkinter import filedialog
import tkinter as tk  # python 3
# import Tkinter as tk  # python 2
from PIL import ImageTk, Image
import model as seg_model
import numpy as np

class GUI(tk.Frame):
    def __init__(self, root):

        tk.Frame.__init__(self, root)

        # toolbar
        toolbar = tk.Frame(root)
        insertBtn = tk.Button(toolbar, text = "Select images", command = self.populate)
        insertBtn.pack(side = tk.LEFT, padx = 4, pady = 4)

        segmentBtn = tk.Button(toolbar, text = "Segment >>", command = self.segment_images)
        segmentBtn.pack(side = tk.LEFT, padx = 4, pady = 4)

        toolbar.pack(side = tk.TOP, fill = tk.X)

        self.canvas = tk.Canvas(root, borderwidth=5, background="#ffffff")
        self.frame = tk.Frame(self.canvas, background="#ffffff")
        self.vsb = tk.Scrollbar(root, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)

        self.vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.create_window((4,4), window=self.frame, anchor="nw", 
                                  tags="self.frame")

        self.frame.bind("<Configure>", self.onFrameConfigure)

        self.image_labels = []

    def populate(self):
        self.destroy()
        initial_dir = "/"
        self.filenames =  filedialog.askopenfilenames(initialdir = initial_dir, title = "Select images",filetypes = (("png files","*.png"),("all files","*.*")))

        print(self.filenames)

        for image_label in self.image_labels:
            image_label.destroy()
        
        self.image_labels = []

        im_names = self.filenames
        for i in range(len(im_names)):
            anotherImg = ImageTk.PhotoImage(Image.open(im_names[i]))
            anotherPanel = tk.Label(self.frame, image = anotherImg)
            anotherPanel.image = anotherImg
            anotherPanel.grid(row = i, column = 0, padx = 20, pady = 20)
            self.image_labels.append(anotherPanel)

    def destroy(self):
        pass

    def round_image(self, image):
        flat_img = image.flatten()
        flat_img = flat_img * 255
        #flat_img_round = [0 if v <= threshold  else 1 for v in flat_img]
        round_img = np.reshape(flat_img, image.shape)
        return round_img

    def round_image_with_threshold(self, image, threshold):
        flat_img = image.flatten()
        flat_img_round = [0 if v <= threshold  else 255 for v in flat_img]
        flat_img_round = np.array(flat_img_round).astype(np.uint8)
        round_img = np.reshape(flat_img_round, image.shape)
        return round_img

    def segment_images(self):
        width = height = 256
        channel = 3
        n = len(self.filenames)
        model = seg_model.create_model(width, height, channel)
        model.load_weights("./model_256x256.h5")

        images_to_predict = np.zeros((len(self.filenames), width, height, channel),dtype = np.uint8)
        masks = []

        for i in range(n):
            img = Image.open(self.filenames[i])
            #new_img = img.resize((width,height))
            np_img = np.array(img)
            np_img = np_img[...,:3]
            images_to_predict[i] = np_img

        results = model.predict(images_to_predict)
        
        i = 0
        for result in results:
            result_i = np.squeeze(result)
            #predictions.append(result_i)
            print(result_i)

            #im = Image.fromarray(result_i)
            #im_rgb = im.convert("RGB")
            #im_rgb.save("result-"+str(i)+".png")
            img =  ImageTk.PhotoImage(image= Image.fromarray(self.round_image_with_threshold(result_i, 0.2)))
            anotherPanel = tk.Label(self.frame, image = img)
            anotherPanel.image = img
            anotherPanel.grid(row = i, column = 1, padx = 20, pady = 20)
            self.image_labels.append(anotherPanel)

            i += 1
                
    def onFrameConfigure(self, event):
        '''Reset the scroll region to encompass the inner frame'''
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

if __name__ == "__main__":
    root=tk.Tk()
    root.title(' CSE-574 Building segmentation from drone imagery')
    root.geometry("600x900") #You want the size of the app to be 500x500
    root.resizable(0, 0)
    GUI(root).pack(side="top", fill="both", expand=True)
    root.mainloop()