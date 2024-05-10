import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedStyle

from LSTM import nnModel
from SVM import svm
from DecisionTree import DecisionTree
window = tk.Tk()
window.title("shamardan")
window.geometry("340x440")
window.configure(bg='#444444')
# Create a themed style
style = ThemedStyle(window)
style.set_theme("equilux")

frame = ttk.Frame(window)
frame.configure(style='Custom.TFrame' , padding=25)
# Creating widgets



Welcome = ttk.Label(frame, text="Welcome To", font=("Arial", 16),padding=(3,6))
ProName = ttk.Label(frame, text="Shamardan", font=("Arial", 22, "bold"), padding=(12, 6) )
Welcome.configure(foreground="#00d7ff")
ProName.configure(foreground="#ffffff",background='#00d7ff')
DecisionTree = ttk.Button(frame,text="Decision Tree",padding=10,command=DecisionTree)
SVM = ttk.Button(frame,text="Support Vector Machine",padding=10,command=svm)
NN = ttk.Button(frame,text="Neural Network",padding=10,command=nnModel)
password_label = ttk.Label(frame, text="Password")


# Placing widgets on the screen
Welcome.grid(row=0, column=0, )
ProName.grid(row=1, column=0,pady=(0, 30))
DecisionTree.grid(row=2, column=0, pady=5)
SVM.grid(row=3, column=0, pady=5)
NN.grid(row=4, column=0, pady=5)



# Configure row and column weights to expand the frame
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

# Center the frame in the window4

frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
window.mainloop()
