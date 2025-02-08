import customtkinter as ctk
from settings import *


class ImageLoader(ctk.CTkFrame):
    def __init__(self, parent, load_image):
        super().__init__(master=parent, fg_color=BLACK)
        self.grid(column=0, row=0, columnspan=2, sticky=ctk.NSEW)
        # DATA.
        self.load_image = load_image
        # WIDGET.
        ctk.CTkButton(
            master=self,
            height=50,
            text="Open Image",
            font=ctk.CTkFont("Cambria", 20, "bold", "italic"),
            command=self.open_dialog,
        ).place(relx=0.5, rely=0.5, anchor=ctk.CENTER)

    def open_dialog(self):
        path = ctk.filedialog.askopenfilename(
            title="Select an image file",
            filetypes=[("Image", "*.png *.jpg *.jpeg")],
        )
        self.load_image(path)


class ImageEditor(ctk.CTkCanvas):
    def __init__(self, parent, resize_image):
        super().__init__(
            master=parent,
            background=CANVAS_BG,
            bd=0,
            highlightthickness=0,
            relief=ctk.RIDGE,
        )
        self.grid(row=0, column=1, sticky=ctk.NSEW, padx=10, pady=10)
        # EVENT.
        self.bind("<Configure>", resize_image)


class CloseEditor(ctk.CTkButton):
    def __init__(self, parent, close_editor):
        super().__init__(
            master=parent,
            width=50,
            height=50,
            text="X",
            text_color=WHITE,
            font=ctk.CTkFont(MAIN_FONT, 20),
            fg_color="transparent",
            hover_color=CLOSE_RED,
            command=close_editor,
        )
        self.place(relx=0.99, rely=0.01, anchor=ctk.NE)

