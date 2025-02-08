import customtkinter as ctk
from os.path import dirname, basename, splitext, join, exists
from PIL import Image, ImageTk, ImageOps, ImageEnhance, ImageFilter
from settings import *
from widgets import *
from menu import Menu
from transformers import CLIPProcessor, CLIPModel
import os
from tkinter import filedialog, messagebox
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load the model and processor
model = None
processor = None

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

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        # SETUP.
        ctk.set_appearance_mode("dark")
        self.geometry("1080x650")
        self.minsize(800, 650)
        self.title("")
        # LAYOUT.
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=3, uniform="A")
        self.columnconfigure(1, weight=7, uniform="A")
        # DATA.
        self.image_width = self.image_height = 0
        self.canvas_width = self.canvas_height = 0
        self.binding_data()
        # WIDGET.
        self.loader = ImageLoader(self, self.load_image)

        # Query and Folder Selection Widgets
        self.query_frame = ctk.CTkFrame(self)
        self.query_frame.grid(row=0, column=0, sticky="nsew")

        self.folder_label = ctk.CTkLabel(self.query_frame, text="Select the Folder:")
        self.folder_label.pack(padx=10, pady=10)  # Add margin or gap

        self.folder_entry = ctk.CTkEntry(self.query_frame, width=300)
        self.folder_entry.pack(padx=10, pady=10)  # Add margin or gap
        self.folder_entry.bind("<Button-1>", lambda e: self.select_folder())

        self.select_folder_button = ctk.CTkButton(self.query_frame, text="Select Folder", command=self.select_folder, width=300)
        self.select_folder_button.pack(padx=10, pady=10)  # Add margin or gap

        self.query_label = ctk.CTkLabel(self.query_frame, text="Describe your photo:")
        self.query_label.pack(padx=10, pady=10)  # Add margin or gap

        self.query_entry = ctk.CTkEntry(self.query_frame, width=300)
        self.query_entry.pack(padx=10, pady=10)  # Add margin or gap

        self.search_button = ctk.CTkButton(self.query_frame, text="Search", command=self.run_similarity, width=300)
        self.search_button.pack(padx=10, pady=10)  # Add margin or gap

        self.thumbnail_frame = ctk.CTkScrollableFrame(self)
        self.thumbnail_frame.grid(row=0, column=1, sticky="nsew")

        self.best_image_frame = ctk.CTkFrame(self)
        self.best_image_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")

        # Loading screen
        self.loading_label = ctk.CTkLabel(self, text="Searching...", font=ctk.CTkFont(MAIN_FONT, 20))

    def load_images(self, folder_path):
        return [join(folder_path, img) for img in os.listdir(folder_path) if img.endswith((".png", ".jpg", ".jpeg"))]

    def get_image_query_similarity(self, query, images_paths):
        images = [Image.open(img_path) for img_path in images_paths]
        inputs = processor(text=[query] * len(images), images=images, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        return logits_per_image.softmax(dim=0).tolist()  # normalize

    def select_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.folder_entry.delete(0, ctk.END)
            self.folder_entry.insert(0, folder_path)
            self.display_thumbnails(folder_path)

    def display_thumbnails(self, folder_path):
        for widget in self.thumbnail_frame.winfo_children():
            widget.destroy()  # Clear any existing thumbnails

        image_paths = self.load_images(folder_path)
        for i, img_path in enumerate(image_paths):
            try:
                img = Image.open(img_path)
                img.thumbnail((150, 150))  # Resize image to thumbnail size
                img_tk = ImageTk.PhotoImage(img)
                frame = ctk.CTkFrame(self.thumbnail_frame, fg_color="white", border_width=1)  # Use highlightthickness instead of border_width
                label = ctk.CTkLabel(frame, image=img_tk, text="", fg_color="white")
                label.image = img_tk  # Keep a reference to avoid garbage collection
                label.pack(expand=True, fill="both", padx=3, pady=3)  # Remove padding
                frame.grid(row=i // 5, column=i % 5, padx=5, pady=5, ipadx=0, ipady=0, sticky="nsew")  # Arrange in a grid (5 per row)
                frame.bind("<Button-1>", lambda e, path=img_path: self.load_image(path))  # Bind click event to open editor
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

    def run_similarity(self):
        query = self.query_entry.get()
        folder_path = self.folder_entry.get()
        if not folder_path:
            messagebox.showerror("Error", "Please select a folder first.")
            return

        # Show loading screen
        self.loading_label.place(relx=0.5, rely=0.5, anchor=ctk.CENTER)
        self.update_idletasks()

        image_paths = self.load_images(folder_path)
        similarity_scores = self.get_image_query_similarity(query, image_paths)
        best_match_index = similarity_scores.index(max(similarity_scores))
        best_image_path = image_paths[best_match_index]

        # Hide loading screen
        self.loading_label.place_forget()

        # Display the best-matching image with the query
        self.display_best_image(best_image_path, query)

    def display_best_image(self, image_path, query):
        global best_image_displayed_path  # To share the image path for saving or editing
        best_image_displayed_path = image_path

        for widget in self.best_image_frame.winfo_children():
            widget.destroy()  # Clear any existing displayed image

        try:
            img = Image.open(image_path)
            img.thumbnail((300, 300))  # Resize image for the display area
            img_tk = ImageTk.PhotoImage(img)
            label = ctk.CTkLabel(self.best_image_frame, image=img_tk, text="", fg_color="white", corner_radius=10)
            label.image = img_tk  # Keep a reference to avoid garbage collection
            label.pack(padx=10, pady=10)

            # Generate image description with increased length and include the query
            inputs = blip_processor(images=img, return_tensors="pt")
            generated_ids = blip_model.generate(**inputs, max_length=200, num_beams=5, early_stopping=True)
            description = blip_processor.decode(generated_ids[0], skip_special_tokens=True)

            # Display the description
            description_label = ctk.CTkLabel(self.best_image_frame, text=description, fg_color="black", wraplength=300)
            description_label.pack(padx=10, pady=10)

        except Exception as e:
            print(f"Error displaying best image: {e}")

        # Load the image into the editor
        self.load_image(image_path)

    def load_image(self, path):
        try:
            self.original = Image.open(path)
        except:
            pass
        else:
            self.image = self.original.copy()
            self.image_tk = ImageTk.PhotoImage(self.image)
            self.image_ratio = self.image.width / self.image.height
            # HIDE THE IMAGE LOADER.
            self.loader.grid_forget()
            # OPEN THE IMAGE EDITOR.
            self.editor = ImageEditor(self, self.resize_image)
            self.menu = Menu(self, self.binding_source, path, self.save_image)
            self.closer = CloseEditor(self, self.close_editor)

    def binding_data(self):
        # BINDING.
        self.binding_source = {
            "POSITION": {
                "ROTATE": ctk.DoubleVar(value=DEFAULT_ROTATE),
                "ZOOM": ctk.DoubleVar(value=DEFAULT_ZOOM),
                "FLIP": ctk.StringVar(value=FLIP_OPTIONS[0]),
            },
            "COLOUR": {
                "BRIGHTNESS": ctk.DoubleVar(value=DEFAULT_BRIGHTNESS),
                "GRAYSCALE": ctk.BooleanVar(value=DEFAULT_GRAYSCALE),
                "INVERT": ctk.BooleanVar(value=DEFAULT_INVERT),
                "VIBRANCE": ctk.DoubleVar(value=DEFAULT_VIBRANCE),
                "SHARPNESS": ctk.DoubleVar(value=DEFAULT_SHARPNESS),
                "CONTRAST": ctk.DoubleVar(value=DEFAULT_COLOR_CONTRAST),
            },
            "EFFECT": {
                "BLUR": ctk.DoubleVar(value=DEFAULT_BLUR),
                "CONTRAST": ctk.IntVar(value=DEFAULT_EFFECT_CONTRAST),
                "EFFECT": ctk.StringVar(value=EFFECT_OPTIONS[0]),
            },
        }
        # TRACING.
        for data_source in self.binding_source.values():
            for binding_data in data_source.values():
                binding_data.trace_add("write", self.manipulate_image)

    def manipulate_image(self, *args):
        self.image = self.original.copy()
        # ROTATE.
        value = self.binding_source["POSITION"]["ROTATE"].get()
        if value != DEFAULT_ROTATE:
            self.image = self.image.rotate(value)
        # ZOOM.
        value = self.binding_source["POSITION"]["ZOOM"].get()
        if value != DEFAULT_ZOOM:
            self.image = ImageOps.crop(self.image, value)
        # FLIP.
        match self.binding_source["POSITION"]["FLIP"].get():
            case "NONE":
                pass
            case "X":
                self.image = ImageOps.mirror(self.image)
            case "Y":
                self.image = ImageOps.flip(self.image)
            case "BOTH":
                self.image = ImageOps.mirror(self.image)
                self.image = ImageOps.flip(self.image)
        # BRIGHTNESS.
        value = self.binding_source["COLOUR"]["BRIGHTNESS"].get()
        if value != DEFAULT_BRIGHTNESS:
            ENHANCER = ImageEnhance.Brightness(self.image)
            self.image = ENHANCER.enhance(value)
        # VIBRANCE.
        value = self.binding_source["COLOUR"]["VIBRANCE"].get()
        if value != DEFAULT_VIBRANCE:
            ENHANCER = ImageEnhance.Color(self.image)
            self.image = ENHANCER.enhance(value)
        # SHARPNESS.
        value = self.binding_source["COLOUR"]["SHARPNESS"].get()
        if value != DEFAULT_SHARPNESS:
            ENHANCER = ImageEnhance.Sharpness(self.image)
            self.image = ENHANCER.enhance(value)
        # COLOR CONTRAST.
        value = self.binding_source["COLOUR"]["CONTRAST"].get()
        if value != DEFAULT_COLOR_CONTRAST:
            ENHANCER = ImageEnhance.Contrast(self.image)
            self.image = ENHANCER.enhance(value)
        # GRAYSCALE.
        if self.binding_source["COLOUR"]["GRAYSCALE"].get():
            self.image = ImageOps.grayscale(self.image)
        # INVERT.
        if self.binding_source["COLOUR"]["INVERT"].get():
            self.image = ImageOps.invert(self.image)
        # BLUR.
        value = self.binding_source["EFFECT"]["BLUR"].get()
        if value != DEFAULT_BLUR:
            self.image = self.image.filter(ImageFilter.GaussianBlur(value))
        # EFFECT CONTRAST.
        value = self.binding_source["EFFECT"]["CONTRAST"].get()
        if value != DEFAULT_EFFECT_CONTRAST:
            self.image = self.image.filter(ImageFilter.UnsharpMask(value))
        # EFFECTS.
        match self.binding_source["EFFECT"]["EFFECT"].get():
            case "NONE":
                pass
            case "EMBOSS":
                self.image = self.image.filter(ImageFilter.EMBOSS)
            case "FIND EDGES":
                self.image = self.image.filter(ImageFilter.FIND_EDGES)
            case "CONTOUR":
                self.image = self.image.filter(ImageFilter.CONTOUR)
            case "EDGE ENHANCE":
                self.image = self.image.filter(ImageFilter.EDGE_ENHANCE_MORE)
            case "BLUR":
                self.image = self.image.filter(ImageFilter.BLUR)
            case "DETAIL":
                self.image = self.image.filter(ImageFilter.DETAIL)
            case "SHARPEN":
                self.image = self.image.filter(ImageFilter.SHARPEN)
            case "SMOOTH":
                self.image = self.image.filter(ImageFilter.SMOOTH_MORE)
        # SHOW IMAGE.
        self.show_image()

    def resize_image(self, event):
        # CURRENT RATIO.
        self.canvas_width = event.width
        self.canvas_height = event.height
        canvas_ratio = self.canvas_width / self.canvas_height
        # GET NEW WIDTH & HEIGHT OF IMAGE.
        if canvas_ratio > self.image_ratio:
            self.image_height = int(self.canvas_height)
            self.image_width = int(self.image_height * self.image_ratio)
        else:
            self.image_width = int(self.canvas_width)
            self.image_height = int(self.image_width / self.image_ratio)
        # SHOW IMAGE.
        self.show_image()

    def show_image(self):
        # CUSTOMIZED IMAGE.
        resized_image = self.image.resize((self.image_width, self.image_height))
        self.image_tk = ImageTk.PhotoImage(resized_image)
        # DISCARD BEFORE DRAW.
        self.editor.delete(ctk.ALL)
        # DRAW NEW IMAGE.
        self.editor.create_image(
            self.canvas_width / 2, self.canvas_height / 2, image=self.image_tk
        )

    def close_editor(self):
        # HIDE THE IMAGE EDITOR.
        self.menu.grid_forget()
        self.editor.grid_forget()
        self.closer.place_forget()
        # RESET TO INITIAL SEARCH STATE.
        self.query_frame.grid(row=0, column=0, sticky="nsew")
        self.thumbnail_frame.grid(row=0, column=1, sticky="nsew")
        self.best_image_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")

    def save_image(self, file_path, file_name, extension):
        # HANDLE NO FILE NAME.
        file_name = file_name if file_name else "default"
        path = f"{join(file_path, file_name)}.{extension}"
        # HANDLE EXISTED FILE.
        counter = 0
        while exists(path):
            counter += 1
            path = f"{join(file_path, file_name)}_{counter}.{extension}"
        # EXPORT IMAGE.
        self.image.save(path)

if __name__ == "__main__":
    # Load the BLIP model and processor
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    # Load the model and processor within the main block
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    App().mainloop()


