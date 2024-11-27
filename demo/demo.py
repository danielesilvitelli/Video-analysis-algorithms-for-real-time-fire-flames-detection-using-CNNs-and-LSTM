import os
import cv2
import copy
import numpy as np
import tkinter as tk
from tkinter import ttk
from collections import deque
from tkinter import filedialog
from keras.models import load_model




class GUI():

    def __init__(self):
        self.filepath = None
        self.frames_to_display = []
        self.fps = None
        self.width = None
        self.height = None
        self.flag = False
    
    def real_time_check(self, choice):
        if choice == '--- select the scenario ---':
            return
        if choice == 'Wild long-range':
            model_path = 'models/wild_long-range_model.h5'
        elif choice == 'Wild short-range':
            model_path = 'models/wild_short-range_model.h5'
        elif choice == 'Urban long-range':
            model_path = 'models/urban_long-range_model.h5'
        else:
            model_path = 'models/live_demo_model.h5'

        model = load_model(model_path)

        # define a video capture object
        vid = cv2.VideoCapture(0)

        sequence = deque()
        temporal_length = 4
        temporal_stride = 2
        label = "fire flames NOT detected"
        temp = False

        while(True):
            ret, frame = vid.read() # Capture the video frame by frame
            if ret == True:
                height, width, _ = frame.shape
                frame = cv2.resize(frame, (224,224))
                sequence.append(frame[:,:, ::-1])
                if len(sequence) == temporal_length:
                    temp = True
                    sequence_c = copy.deepcopy(sequence)
                    for t in range(temporal_stride):
                        sequence.popleft()
                    sequence_c = [list(sequence_c)]
                    sequence_c = np.array(sequence_c)
                    pred = model.predict(sequence_c)
                    label = "fire flames detected " + str(round(pred[0][0]*100, 3)) + "%" if pred[0][0] > 0.7 else "fire flames NOT detected " + str(round(pred[0][0]*100, 3)) + "%"
                if temp:
                    frame = cv2.resize(frame, (width, height))
                    # Draw the predicted label on the frame
                    color = (30, 180, 30) if "NOT" in label else (0, 0, 255)
                    frame_to_display = cv2.putText(
                                                    img=frame, 
                                                    text=label, 
                                                    org=(50, 50), 
                                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                                                    fontScale=1, 
                                                    color=color,
                                                    thickness=2
                                                )
                    # Create a named window with the desired size
                    cv2.namedWindow("Real-time video analysis", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Real-time video analysis", (1000, 600))
                    cv2.moveWindow("Real-time video analysis", 500, 0)
                    # Display the processed frame
                    cv2.imshow("Real-time video analysis", frame_to_display)
            else: 
                break
            if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'q' to quit
                break
        # After the loop release the cap object
        vid.release()
        # Destroy all the windows
        cv2.destroyAllWindows()

    def browse_file(self):
        fp = filedialog.askopenfilename()
        if fp[-3:] != 'avi' and fp[-3:] != 'mp4' and fp[-3:] != 'MP4' and fp[-3:] != 'MOV':
            self.filepath = None
            return
        self.filepath = fp    

    def play_video(self, choice):
        if choice == '--- select the scenario ---' or self.filepath == None:
            return
        if choice == 'Wild long-range':
            model_path = 'models/wild_long-range_model.h5'
        elif choice == 'Wild short-range':
            model_path = 'models/wild_short-range_model.h5'
        elif choice == 'Urban long-range':
            model_path = 'models/urban_long-range_model.h5'
        else:
            model_path = 'models/urban_short-range_model.h5'

        model = load_model(model_path)

        cap = cv2.VideoCapture(self.filepath)
        # Get info about video capture
        self.fps = int(round(cap.get(cv2.CAP_PROP_FPS), 0))
        self.width = int(cap.get(3))
        self.height = int(cap.get(4))

        sequence = deque()
        temporal_length = 4
        temporal_stride = 2
        label = "fire flames NOT detected"
        temp = False

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < temporal_length:
            print("Video too short. Minimum number of frames must be 4.")

        self.frames_to_display = []

        # Extracting frames from video        
        if (cap.isOpened() == False): 
            print("Error opening video stream or file")
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                frame = cv2.resize(frame, (224,224))
                sequence.append(frame[:,:, ::-1])
                if len(sequence) == temporal_length:
                    temp = True
                    sequence_c = copy.deepcopy(sequence)
                    for t in range(temporal_stride):
                        sequence.popleft()
                    sequence_c = [list(sequence_c)]
                    sequence_c = np.array(sequence_c)
                    pred = model.predict(sequence_c)
                    label = "fire flames detected " + str(round(pred[0][0]*100, 3)) + "%" if pred[0][0] > 0.8 else "fire flames NOT detected " + str(round(pred[0][0]*100, 3)) + "%"
                if temp:
                    frame = cv2.resize(frame, (1000, 600))
                    # Draw the predicted label on the frame
                    color = (30, 180, 30) if "NOT" in label else (0, 0, 255)
                    frame_to_display = cv2.putText(
                                                    img=frame, 
                                                    text=label, 
                                                    org=(50, 50), 
                                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                                                    fontScale=1, 
                                                    color=color,
                                                    thickness=2
                                                )
                    self.frames_to_display.append(frame_to_display)
                    # Create a named window with the desired size
                    cv2.namedWindow("Video analysis", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Video analysis", (1000, 600))
                    cv2.moveWindow("Video analysis", 500, 0)
                    # Display the processed frame
                    cv2.imshow("Video analysis", frame_to_display)
            else: 
                break
            if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'q' to quit
                break
        cap.release()
        cv2.destroyAllWindows()

        self.flag = True         
    
    def save_video(self):
        if self.filepath == None or self.flag == False:
            return
        rev = self.filepath[::-1]
        till_slash = rev[:rev.index('/')]
        video_name = till_slash[::-1]
        # Saving the video in the proper folder (we take the video name)
        out = cv2.VideoWriter(os.getcwd() + '/' + video_name + '_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (1000, 600))

        for frame in self.frames_to_display:
            out.write(frame)
        out.release()
        print('\n\nVideo saved at: ' + os.getcwd() + '/' + video_name + '_output.mp4')



root = tk.Tk()
root.configure(background='dark blue') # Apply the custom style to the root window
style = ttk.Style() # Create a ttk.Style object
screen_width = 300
screen_height = 300
root.geometry(f"{screen_width}x{screen_height}") # set the geometry of the window to be full-screen

# create a list of options for the drop-down menu
options = ["--- select the scenario ---", "Urban long-range", "Urban short-range", "Wild long-range", "Wild short-range",]
# create a StringVar to store the selected option
selected_option = tk.StringVar(root)
selected_option.set('--- select the scenario ---')  # set the initial selected option
# create the drop-down menu
option_menu = tk.OptionMenu(root, selected_option, *options)
option_menu.pack()

def enable_button(*args):
    if selected_option.get() != "--- select the scenario ---":
        rt_button.config(state="normal")
        upload_button.config(state="normal")
        play_button.config(state="normal")
        save_button.config(state="normal")
    else:
        rt_button.config(state="disabled")
        upload_button.config(state="disabled")
        play_button.config(state="disabled")
        save_button.config(state="disabled")

selected_option.trace("w", enable_button)

# Define a custom style for a TButton widget
style.configure('Custom.TButton', background='red', foreground='black', font=('Times New Roman', 14))
style.configure('RealTime.TButton', background='green', foreground='black', font=('Times New Roman', 14))
style.configure('Upload.TButton', background='yellow', foreground='black', font=('Times New Roman', 14))
style.configure('Save.TButton', background='gray', foreground='black', font=('Times New Roman', 14))

gui = GUI()

# Create TButton widgets with the custom style
rt_button = ttk.Button(root, text="Open camera", style='RealTime.TButton', state='disabled', command=lambda: gui.real_time_check(selected_option.get()))
rt_button.pack()
upload_button = ttk.Button(root, text="Upload Video", style='Upload.TButton', state='disabled', command=lambda: gui.browse_file())
upload_button.pack()
play_button = ttk.Button(root, text="Play", style='Custom.TButton', state='disabled', command=lambda: gui.play_video(selected_option.get()))
play_button.pack()
save_button = ttk.Button(root, text="Save Video", style='Save.TButton', state='disabled', command=lambda: gui.save_video())
save_button.pack()

root.mainloop()
