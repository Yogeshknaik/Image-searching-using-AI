import customtkinter as ctk
from os.path import dirname, basename, splitext
from panels import *


class Menu(ctk.CTkTabview):
    def __init__(self, parent, binding_source, image_path, save_image):
        super().__init__(master=parent)
        self.grid(column=0, row=0, sticky=ctk.NSEW, padx=10, pady=10)
        # TABS.
        self.add("POSITION")
        self.add("COLOUR")
        self.add("EFFECT")
        self.add("EXPORT")
        # FRAMES.
        PositionFrame(self.tab("POSITION"), binding_source["POSITION"])
        ColourFrame(self.tab("COLOUR"), binding_source["COLOUR"])
        EffectFrame(self.tab("EFFECT"), binding_source["EFFECT"])
        ExportFrame(self.tab("EXPORT"), image_path, save_image)


class PositionFrame(ctk.CTkFrame):
    def __init__(self, parent, data_source):
        super().__init__(master=parent, fg_color="transparent")
        self.pack(expand=ctk.TRUE, fill=ctk.BOTH)
        # WIDGETS.
        SliderPanel(self, "ROTATION", 0, 360, data_source["ROTATE"])
        SliderPanel(self, "ZOOM", 0,300, data_source["ZOOM"])
        SegmentPanel(self, "FLIP", FLIP_OPTIONS, data_source["FLIP"])
        ResetButton(
            self,
            (data_source["ROTATE"], DEFAULT_ROTATE),
            (data_source["ZOOM"], DEFAULT_ZOOM),
            (data_source["FLIP"], FLIP_OPTIONS[0]),
        )


class ColourFrame(ctk.CTkFrame):
    def __init__(self, parent, data_source):
        super().__init__(master=parent, fg_color="transparent")
        self.pack(expand=ctk.TRUE, fill=ctk.BOTH)
        # WIDGETS.
        SwitchPanel(
            self,
            ("B/W", data_source["GRAYSCALE"]),
            ("INVERT", data_source["INVERT"]),
        )
        SliderPanel(self, "BRIGHTNESS", 0, 5, data_source["BRIGHTNESS"])
        SliderPanel(self, "VIBRANCE", 0, 5, data_source["VIBRANCE"])
        SliderPanel(self, "SHARPNESS", 0, 5, data_source["SHARPNESS"])
        SliderPanel(self, "CONTRAST", 0, 5, data_source["CONTRAST"])
        ResetButton(
            self,
            (data_source["GRAYSCALE"], DEFAULT_GRAYSCALE),
            (data_source["INVERT"], DEFAULT_INVERT),
            (data_source["BRIGHTNESS"], DEFAULT_BRIGHTNESS),
            (data_source["VIBRANCE"], DEFAULT_VIBRANCE),
            (data_source["SHARPNESS"], DEFAULT_SHARPNESS),
            (data_source["CONTRAST"], DEFAULT_COLOR_CONTRAST),
        )


class EffectFrame(ctk.CTkFrame):
    def __init__(self, parent, data_source):
        super().__init__(master=parent, fg_color="transparent")
        self.pack(expand=ctk.TRUE, fill=ctk.BOTH)
        # WIDGETS.
        DropDownPanel(self, EFFECT_OPTIONS, data_source["EFFECT"])
        SliderPanel(self, "BLUR", 0, 3, data_source["BLUR"])
        SliderPanel(self, "CONTRAST", 0, 10, data_source["CONTRAST"])
        ResetButton(
            self,
            (data_source["EFFECT"], EFFECT_OPTIONS[0]),
            (data_source["BLUR"], DEFAULT_BLUR),
            (data_source["CONTRAST"], DEFAULT_EFFECT_CONTRAST),
        )


class ExportFrame(ctk.CTkFrame):
    def __init__(self, parent, image_path, save_image):
        super().__init__(master=parent, fg_color="transparent")
        self.pack(expand=ctk.TRUE, fill=ctk.BOTH)
        # PATH PROCESS.
        path_name = dirname(image_path)
        file_name, extension = splitext(basename(image_path))
        file_name += "_PROCESSED"
        extension = extension.removeprefix(".")
        # DATA CONTROL.
        self.binding_file_name = ctk.StringVar(value=file_name)
        self.binding_extension = ctk.StringVar(value=extension)
        self.binding_file_path = ctk.StringVar(value=path_name)
        # WIDGET.
        FileNamePanel(self, self.binding_file_name, self.binding_extension)
        FilePathPanel(self, self.binding_file_path)
        SaveButton(
            self,
            self.binding_file_path,
            self.binding_file_name,
            self.binding_extension,
            save_image,
        )
