import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import threading
from queue import Queue

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Aplikasi Pengolahan Citra Digital")
        self.root.geometry("1300x850")
        self.root.configure(bg='#2b2b2b')
        
        self.original_image = None
        self.processed_image = None
        self.original_cv = None
        self.processed_cv = None
        self.processing = False  # Flag untuk menandai sedang processing
        
        self.setup_ui()
        
    def setup_ui(self):
        # Header
        header = tk.Frame(self.root, bg='#4a90e2', height=70)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(header, text="APLIKASI PENGOLAHAN CITRA DIGITAL", 
                font=('Arial', 16, 'bold'), bg='#4a90e2', fg='white').pack(expand=True)
        tk.Label(header, text="Image Processing Application", 
                font=('Arial', 9), bg='#4a90e2', fg='#d0e4ff').pack()
        
        # Progress bar (hidden initially)
        self.progress_frame = tk.Frame(self.root, bg='#2b2b2b')
        self.progress_bar = ttk.Progressbar(self.progress_frame, mode='indeterminate', length=300)
        self.progress_label = tk.Label(self.progress_frame, text="Processing...", bg='#2b2b2b', fg='white')
        
        # Main container
        main = tk.Frame(self.root, bg='#2b2b2b')
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Image display area
        img_frame = tk.Frame(main, bg='#2b2b2b')
        img_frame.pack(fill=tk.X, pady=10)
        
        # Original image
        left = tk.Frame(img_frame, bg='#2b2b2b')
        left.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5)
        tk.Label(left, text="GAMBAR ORIGINAL", font=('Arial', 10, 'bold'), 
                bg='#2b2b2b', fg='#4a90e2').pack()
        self.original_canvas = tk.Canvas(left, width=400, height=400, bg='#1e1e1e', 
                                         highlightthickness=1, highlightbackground='#4a90e2')
        self.original_canvas.pack()
        
        # Processed image
        right = tk.Frame(img_frame, bg='#2b2b2b')
        right.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=5)
        tk.Label(right, text="GAMBAR HASIL", font=('Arial', 10, 'bold'), 
                bg='#2b2b2b', fg='#4a90e2').pack()
        self.result_canvas = tk.Canvas(right, width=400, height=400, bg='#1e1e1e',
                                       highlightthickness=1, highlightbackground='#4a90e2')
        self.result_canvas.pack()
        
        # Control buttons
        btn_frame = tk.Frame(main, bg='#2b2b2b')
        btn_frame.pack(pady=10)
        
        btn_style = {'font': ('Arial', 10, 'bold'), 'bg': '#4a90e2', 'fg': 'white',
                     'padx': 20, 'pady': 5, 'cursor': 'hand2', 'bd': 0, 'relief': tk.RAISED}
        
        tk.Button(btn_frame, text="📂 LOAD GAMBAR", command=self.load_image, **btn_style).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="🔄 RESET", command=self.reset_image, **btn_style).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="💾 SAVE HASIL", command=self.save_image, **btn_style).pack(side=tk.LEFT, padx=5)
        
        # Notebook (tabs)
        notebook = ttk.Notebook(main)
        notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create tabs
        self.create_transform_tab(notebook)
        self.create_enhancement_tab(notebook)
        self.create_histogram_tab(notebook)
        self.create_noise_tab(notebook)
        self.create_fft_tab(notebook)
        
    def show_progress(self, show=True, text="Processing..."):
        """Tampilkan atau sembunyikan progress bar"""
        if show:
            self.progress_label.config(text=text)
            self.progress_frame.pack(pady=5)
            self.progress_bar.start(10)
            self.root.update()
        else:
            self.progress_bar.stop()
            self.progress_frame.pack_forget()
            self.root.update()
        
    def create_transform_tab(self, notebook):
        tab = tk.Frame(notebook, bg='#2b2b2b')
        notebook.add(tab, text="📐 Transformasi Geometri")
        
        # Create scrollable frame
        canvas = tk.Canvas(tab, bg='#2b2b2b', highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollable = tk.Frame(canvas, bg='#2b2b2b')
        
        scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable, anchor="nw", width=900)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Rotasi
        frame1 = tk.LabelFrame(scrollable, text="Rotasi", bg='#3c3c3c', fg='white', font=('Arial', 10, 'bold'))
        frame1.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(frame1, text="Sudut:", bg='#3c3c3c', fg='white').grid(row=0, column=0, padx=10, pady=10)
        self.rotation_var = tk.IntVar(value=0)
        scale = tk.Scale(frame1, from_=0, to=360, orient=tk.HORIZONTAL, variable=self.rotation_var,
                        bg='#3c3c3c', fg='white', length=300)
        scale.grid(row=0, column=1, padx=10, pady=10)
        self.rot_label = tk.Label(frame1, text="0°", bg='#3c3c3c', fg='#4a90e2', font=('Arial', 10, 'bold'))
        self.rot_label.grid(row=0, column=2, padx=10)
        scale.configure(command=lambda x: self.rot_label.config(text=f"{int(float(x))}°"))
        
        tk.Button(frame1, text="Apply Rotasi", command=lambda: self.apply_geometric("rotate"),
                 bg='#4a90e2', fg='white', padx=15, pady=5).grid(row=0, column=3, padx=20)
        
        # Scaling
        frame2 = tk.LabelFrame(scrollable, text="Scaling (Perbesar/Perkecil)", bg='#3c3c3c', fg='white', font=('Arial', 10, 'bold'))
        frame2.pack(fill=tk.X, padx=20, pady=10)
        
        # Info label
        info_label = tk.Label(frame2, text="💡 Tips: Gunakan nilai antara 0.5 - 2.0 untuk hasil optimal", 
                             bg='#3c3c3c', fg='#ffaa00', font=('Arial', 9, 'italic'))
        info_label.grid(row=0, column=0, columnspan=4, padx=10, pady=5)
        
        tk.Label(frame2, text="Scale X:", bg='#3c3c3c', fg='white').grid(row=1, column=0, padx=10, pady=10)
        self.scale_x = tk.DoubleVar(value=1.0)
        scale_x_spin = tk.Spinbox(frame2, from_=0.1, to=3.0, increment=0.1, textvariable=self.scale_x,
                                width=10, bg='#2b2b2b', fg='white')
        scale_x_spin.grid(row=1, column=1, padx=10)
        
        tk.Label(frame2, text="Scale Y:", bg='#3c3c3c', fg='white').grid(row=2, column=0, padx=10, pady=10)
        self.scale_y = tk.DoubleVar(value=1.0)
        scale_y_spin = tk.Spinbox(frame2, from_=0.1, to=3.0, increment=0.1, textvariable=self.scale_y,
                                width=10, bg='#2b2b2b', fg='white')
        scale_y_spin.grid(row=2, column=1, padx=10)
        
        # Preset buttons
        preset_frame = tk.Frame(frame2, bg='#3c3c3c')
        preset_frame.grid(row=1, column=2, rowspan=2, padx=10)
        
        tk.Button(preset_frame, text="0.5x", command=lambda: self.set_scale(0.5, 0.5),
                 bg='#5a5a5a', fg='white', width=5).pack(side=tk.LEFT, padx=2)
        tk.Button(preset_frame, text="1.0x", command=lambda: self.set_scale(1.0, 1.0),
                 bg='#5a5a5a', fg='white', width=5).pack(side=tk.LEFT, padx=2)
        tk.Button(preset_frame, text="1.5x", command=lambda: self.set_scale(1.5, 1.5),
                 bg='#5a5a5a', fg='white', width=5).pack(side=tk.LEFT, padx=2)
        tk.Button(preset_frame, text="2.0x", command=lambda: self.set_scale(2.0, 2.0),
                 bg='#5a5a5a', fg='white', width=5).pack(side=tk.LEFT, padx=2)
        
        # Quality option
        self.fast_resize = tk.BooleanVar(value=True)
        tk.Checkbutton(frame2, text="Fast Resize (Recommended)", variable=self.fast_resize,
                      bg='#3c3c3c', fg='white', selectcolor='#3c3c3c').grid(row=3, column=0, columnspan=2, padx=10, pady=5)
        
        apply_btn = tk.Button(frame2, text="Apply Scaling", command=self.apply_scaling_safe,
                             bg='#4a90e2', fg='white', padx=20, pady=8, font=('Arial', 10, 'bold'))
        apply_btn.grid(row=1, column=3, rowspan=2, padx=20)
        
        # Flip
        frame3 = tk.LabelFrame(scrollable, text="Flip", bg='#3c3c3c', fg='white', font=('Arial', 10, 'bold'))
        frame3.pack(fill=tk.X, padx=20, pady=10)
        
        btn_frame = tk.Frame(frame3, bg='#3c3c3c')
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="Flip Horizontal", command=lambda: self.apply_geometric("flip_h"),
                 bg='#5a5a5a', fg='white', padx=15, pady=5).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="Flip Vertikal", command=lambda: self.apply_geometric("flip_v"),
                 bg='#5a5a5a', fg='white', padx=15, pady=5).pack(side=tk.LEFT, padx=10)
        
    def set_scale(self, x, y):
        """Set preset scale values"""
        self.scale_x.set(x)
        self.scale_y.set(y)
        
    def apply_scaling_safe(self):
        """Apply scaling with progress indicator and optimization"""
        if self.processed_cv is None:
            messagebox.showwarning("Warning", "⚠️ Load gambar terlebih dahulu!")
            return
        
        if self.processing:
            messagebox.showwarning("Warning", "⚠️ Sedang memproses, tunggu sebentar...")
            return
        
        sx = self.scale_x.get()
        sy = self.scale_y.get()
        
        # Validasi input
        if sx <= 0 or sy <= 0:
            messagebox.showwarning("Warning", "⚠️ Nilai scaling harus lebih dari 0!")
            return
        
        # Batasi scaling maksimal untuk mencegah freeze
        max_scale = 5.0
        if sx > max_scale or sy > max_scale:
            if not messagebox.askyesno("Konfirmasi", f"Nilai scaling {sx}x{sy} cukup besar.\nIni mungkin akan memakan waktu dan memori.\nLanjutkan?"):
                return
        
        # Tampilkan progress
        self.show_progress(True, f"Scaling gambar {sx:.1f}x{sy:.1f}...")
        self.processing = True
        
        # Gunakan after untuk update GUI
        self.root.after(100, lambda: self._do_scaling(sx, sy))
        
    def _do_scaling(self, sx, sy):
        """Lakukan scaling dengan optimasi"""
        try:
            img = self.processed_cv.copy()
            h, w = img.shape[:2]
            new_w, new_h = int(w * sx), int(h * sy)
            
            # Batasi ukuran maksimal untuk mencegah memory error
            max_size = 3000
            if new_w > max_size or new_h > max_size:
                scale_factor = min(max_size / new_w, max_size / new_h)
                new_w = int(new_w * scale_factor)
                new_h = int(new_h * scale_factor)
                messagebox.showwarning("Warning", f"Ukuran terlalu besar, diskalakan menjadi {new_w}x{new_h}")
            
            # Pilih metode resize
            if self.fast_resize.get():
                # Fast resize (lebih cepat)
                result = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                # High quality resize (lebih lambat tapi lebih bagus)
                result = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            self.processed_cv = result
            self.processed_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            self.update_displays()
            
            messagebox.showinfo("Info", f"✅ Scaling berhasil!\nUkuran baru: {new_w} x {new_h}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Gagal melakukan scaling:\n{str(e)}")
        finally:
            self.processing = False
            self.show_progress(False)
        
    def apply_geometric(self, op):
        if self.processed_cv is None:
            messagebox.showwarning("Warning", "⚠️ Load gambar terlebih dahulu!")
            return
        
        if self.processing:
            return
            
        self.show_progress(True, "Memproses...")
        self.processing = True
        
        # Gunakan after untuk mencegah freeze
        self.root.after(100, lambda: self._do_geometric(op))
        
    def _do_geometric(self, op):
        try:
            if op == "rotate":
                angle = self.rotation_var.get()
                h, w = self.processed_cv.shape[:2]
                M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
                result = cv2.warpAffine(self.processed_cv, M, (w, h))
            elif op == "flip_h":
                result = cv2.flip(self.processed_cv, 1)
            elif op == "flip_v":
                result = cv2.flip(self.processed_cv, 0)
            else:
                result = self.processed_cv.copy()
            
            self.processed_cv = result
            self.processed_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            self.update_displays()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error: {str(e)}")
        finally:
            self.processing = False
            self.show_progress(False)
        
    def create_enhancement_tab(self, notebook):
        tab = tk.Frame(notebook, bg='#2b2b2b')
        notebook.add(tab, text="✨ Enhancement")
        
        # Scrollable frame
        canvas = tk.Canvas(tab, bg='#2b2b2b', highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollable = tk.Frame(canvas, bg='#2b2b2b')
        
        scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable, anchor="nw", width=900)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Brightness
        frame1 = tk.LabelFrame(scrollable, text="Brightness", bg='#3c3c3c', fg='white', font=('Arial', 10, 'bold'))
        frame1.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(frame1, text="Nilai:", bg='#3c3c3c', fg='white').grid(row=0, column=0, padx=10, pady=10)
        self.brightness = tk.IntVar(value=0)
        scale = tk.Scale(frame1, from_=-100, to=100, orient=tk.HORIZONTAL, variable=self.brightness,
                        bg='#3c3c3c', fg='white', length=300)
        scale.grid(row=0, column=1, padx=10, pady=10)
        self.bright_label = tk.Label(frame1, text="0", bg='#3c3c3c', fg='#4a90e2')
        self.bright_label.grid(row=0, column=2, padx=10)
        scale.configure(command=lambda x: self.bright_label.config(text=f"{int(float(x))}"))
        
        tk.Button(frame1, text="Apply", command=lambda: self.apply_enhancement("brightness"),
                 bg='#4a90e2', fg='white', padx=15, pady=5).grid(row=0, column=3, padx=20)
        
        # Contrast
        frame2 = tk.LabelFrame(scrollable, text="Contrast", bg='#3c3c3c', fg='white', font=('Arial', 10, 'bold'))
        frame2.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(frame2, text="Nilai:", bg='#3c3c3c', fg='white').grid(row=0, column=0, padx=10, pady=10)
        self.contrast = tk.DoubleVar(value=1.0)
        scale = tk.Scale(frame2, from_=0.5, to=3.0, resolution=0.1, orient=tk.HORIZONTAL, variable=self.contrast,
                        bg='#3c3c3c', fg='white', length=300)
        scale.grid(row=0, column=1, padx=10, pady=10)
        self.cont_label = tk.Label(frame2, text="1.0", bg='#3c3c3c', fg='#4a90e2')
        self.cont_label.grid(row=0, column=2, padx=10)
        scale.configure(command=lambda x: self.cont_label.config(text=f"{float(x):.1f}"))
        
        tk.Button(frame2, text="Apply", command=lambda: self.apply_enhancement("contrast"),
                 bg='#4a90e2', fg='white', padx=15, pady=5).grid(row=0, column=3, padx=20)
        
        # Quick Actions
        frame3 = tk.LabelFrame(scrollable, text="Quick Actions", bg='#3c3c3c', fg='white', font=('Arial', 10, 'bold'))
        frame3.pack(fill=tk.X, padx=20, pady=10)
        
        actions = [("Histogram Equalization", "equalize"), ("Grayscale", "grayscale"),
                   ("Sharpening", "sharpen"), ("Gamma Correction", "gamma")]
        
        for i, (text, cmd) in enumerate(actions):
            btn = tk.Button(frame3, text=text, command=lambda c=cmd: self.apply_enhancement(c),
                           bg='#5a5a5a', fg='white', padx=15, pady=5)
            btn.grid(row=i//2, column=i%2, padx=10, pady=10, sticky='ew')
        frame3.columnconfigure(0, weight=1)
        frame3.columnconfigure(1, weight=1)
        
    def create_histogram_tab(self, notebook):
        tab = tk.Frame(notebook, bg='#2b2b2b')
        notebook.add(tab, text="📊 Histogram")
        
        # Figure
        self.hist_figure = plt.Figure(figsize=(9, 6), dpi=100, facecolor='#1e1e1e')
        self.hist_canvas = FigureCanvasTkAgg(self.hist_figure, tab)
        self.hist_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Buttons
        btn_frame = tk.Frame(tab, bg='#2b2b2b')
        btn_frame.pack(pady=10)
        
        btn_style = {'bg': '#4a90e2', 'fg': 'white', 'padx': 15, 'pady': 5, 'cursor': 'hand2', 'font': ('Arial', 10)}
        tk.Button(btn_frame, text="Histogram Original", command=self.show_histogram_original, **btn_style).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Histogram Hasil", command=self.show_histogram_processed, **btn_style).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Bandingkan", command=self.show_both_histograms, **btn_style).pack(side=tk.LEFT, padx=5)
        
        # Initial placeholder
        self.show_histogram_placeholder()
        
    def create_noise_tab(self, notebook):
        tab = tk.Frame(notebook, bg='#2b2b2b')
        notebook.add(tab, text="🎲 Noise & Filtering")
        
        # Scrollable frame
        canvas = tk.Canvas(tab, bg='#2b2b2b', highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollable = tk.Frame(canvas, bg='#2b2b2b')
        
        scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable, anchor="nw", width=900)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Noise
        frame1 = tk.LabelFrame(scrollable, text="Noise Generator", bg='#3c3c3c', fg='white', font=('Arial', 10, 'bold'))
        frame1.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(frame1, text="Intensitas:", bg='#3c3c3c', fg='white').grid(row=0, column=0, padx=10, pady=10)
        self.noise_intensity = tk.DoubleVar(value=0.05)
        scale = tk.Scale(frame1, from_=0.01, to=0.5, resolution=0.01, orient=tk.HORIZONTAL,
                        variable=self.noise_intensity, bg='#3c3c3c', fg='white', length=300)
        scale.grid(row=0, column=1, padx=10, pady=10)
        self.noise_label = tk.Label(frame1, text="0.05", bg='#3c3c3c', fg='#4a90e2')
        self.noise_label.grid(row=0, column=2, padx=10)
        scale.configure(command=lambda x: self.noise_label.config(text=f"{float(x):.2f}"))
        
        noise_btn = tk.Frame(frame1, bg='#3c3c3c')
        noise_btn.grid(row=1, column=0, columnspan=4, pady=10)
        
        noises = [("Gaussian", "gaussian"), ("Salt & Pepper", "salt_pepper"), ("Speckle", "speckle")]
        for text, cmd in noises:
            tk.Button(noise_btn, text=text, command=lambda c=cmd: self.add_noise(c),
                     bg='#d9534f', fg='white', padx=15, pady=5).pack(side=tk.LEFT, padx=5)
        
        # Filter
        frame2 = tk.LabelFrame(scrollable, text="Image Filter", bg='#3c3c3c', fg='white', font=('Arial', 10, 'bold'))
        frame2.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(frame2, text="Kernel Size:", bg='#3c3c3c', fg='white').grid(row=0, column=0, padx=10, pady=10)
        self.kernel = tk.IntVar(value=3)
        tk.Spinbox(frame2, from_=3, to=15, increment=2, textvariable=self.kernel,
                  width=10, bg='#2b2b2b', fg='white').grid(row=0, column=1, padx=10)
        
        filter_btn = tk.Frame(frame2, bg='#3c3c3c')
        filter_btn.grid(row=1, column=0, columnspan=3, pady=10)
        
        filters = [("Average", "average"), ("Gaussian", "gaussian"), ("Median", "median"), ("Bilateral", "bilateral")]
        for text, cmd in filters:
            tk.Button(filter_btn, text=text, command=lambda c=cmd: self.apply_filter(c),
                     bg='#5bc0de', fg='white', padx=15, pady=5).pack(side=tk.LEFT, padx=5)
        
    def create_fft_tab(self, notebook):
        tab = tk.Frame(notebook, bg='#2b2b2b')
        notebook.add(tab, text="🌊 FFT Analysis")
        
        # Figure
        self.fft_figure = plt.Figure(figsize=(10, 8), dpi=100, facecolor='#1e1e1e')
        self.fft_canvas = FigureCanvasTkAgg(self.fft_figure, tab)
        self.fft_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Buttons
        btn_frame = tk.Frame(tab, bg='#2b2b2b')
        btn_frame.pack(pady=10)
        
        btn_style = {'bg': '#4a90e2', 'fg': 'white', 'padx': 15, 'pady': 5, 'cursor': 'hand2', 'font': ('Arial', 10)}
        tk.Button(btn_frame, text="FFT Original", command=self.analyze_fft_original, **btn_style).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="FFT Hasil", command=self.analyze_fft_processed, **btn_style).pack(side=tk.LEFT, padx=5)
        
        # Low Pass Filter
        filter_frame = tk.LabelFrame(tab, text="Low Pass Filter (Domain Frekuensi)", bg='#3c3c3c', fg='white', font=('Arial', 10, 'bold'))
        filter_frame.pack(fill=tk.X, padx=20, pady=10)
        
        cutoff_frame = tk.Frame(filter_frame, bg='#3c3c3c')
        cutoff_frame.pack(pady=10)
        
        tk.Label(cutoff_frame, text="Cutoff Frequency:", bg='#3c3c3c', fg='white').pack(side=tk.LEFT, padx=10)
        self.cutoff = tk.DoubleVar(value=0.3)
        scale = tk.Scale(cutoff_frame, from_=0.05, to=0.95, resolution=0.05, orient=tk.HORIZONTAL,
                        variable=self.cutoff, bg='#3c3c3c', fg='white', length=300)
        scale.pack(side=tk.LEFT, padx=10)
        self.cutoff_label = tk.Label(cutoff_frame, text="0.30", bg='#3c3c3c', fg='#4a90e2', font=('Arial', 10, 'bold'))
        self.cutoff_label.pack(side=tk.LEFT, padx=10)
        scale.configure(command=lambda x: self.cutoff_label.config(text=f"{float(x):.2f}"))
        
        tk.Button(filter_frame, text="Apply Low Pass Filter", command=self.apply_fft_filter,
                 bg='#5cb85c', fg='white', padx=20, pady=5).pack(pady=10)
        
        # Initial placeholder
        self.show_fft_placeholder()
        
    def show_histogram_placeholder(self):
        """Tampilkan placeholder jika belum ada gambar"""
        self.hist_figure.clear()
        ax = self.hist_figure.add_subplot(111)
        ax.text(0.5, 0.5, '📸 Belum ada gambar\nKlik "LOAD GAMBAR" terlebih dahulu', 
                ha='center', va='center', fontsize=14, color='gray', transform=ax.transAxes)
        ax.set_facecolor('#1e1e1e')
        ax.axis('off')
        self.hist_canvas.draw()
        
    def show_fft_placeholder(self):
        """Tampilkan placeholder jika belum ada gambar"""
        self.fft_figure.clear()
        ax = self.fft_figure.add_subplot(111)
        ax.text(0.5, 0.5, '📸 Belum ada gambar\nKlik "LOAD GAMBAR" terlebih dahulu', 
                ha='center', va='center', fontsize=14, color='gray', transform=ax.transAxes)
        ax.set_facecolor('#1e1e1e')
        ax.axis('off')
        self.fft_canvas.draw()
        
    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            self.show_progress(True, "Loading gambar...")
            try:
                self.original_image = Image.open(file_path).convert('RGB')
                self.original_cv = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_RGB2BGR)
                self.processed_image = self.original_image.copy()
                self.processed_cv = self.original_cv.copy()
                self.update_displays()
                
                # Update histogram and FFT with the loaded image
                self.show_histogram_original()
                self.analyze_fft_original()
                
                messagebox.showinfo("Info", f"✅ Gambar berhasil di-load!\n📏 Ukuran: {self.original_cv.shape[1]}x{self.original_cv.shape[0]}")
            except Exception as e:
                messagebox.showerror("Error", f"Gagal load gambar:\n{str(e)}")
            finally:
                self.show_progress(False)
        
    def apply_enhancement(self, op):
        if self.processed_cv is None:
            messagebox.showwarning("Warning", "⚠️ Load gambar terlebih dahulu!")
            return
        
        if self.processing:
            return
            
        self.show_progress(True, f"Applying {op}...")
        self.processing = True
        self.root.after(100, lambda: self._do_enhancement(op))
        
    def _do_enhancement(self, op):
        try:
            if op == "brightness":
                val = self.brightness.get()
                result = cv2.convertScaleAbs(self.processed_cv, alpha=1, beta=val)
            elif op == "contrast":
                alpha = self.contrast.get()
                result = cv2.convertScaleAbs(self.processed_cv, alpha=alpha, beta=0)
            elif op == "equalize":
                if len(self.processed_cv.shape) == 3:
                    ycrcb = cv2.cvtColor(self.processed_cv, cv2.COLOR_BGR2YCrCb)
                    ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
                    result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
                else:
                    result = cv2.equalizeHist(self.processed_cv)
            elif op == "grayscale":
                result = cv2.cvtColor(self.processed_cv, cv2.COLOR_BGR2GRAY)
                result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            elif op == "sharpen":
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                result = cv2.filter2D(self.processed_cv, -1, kernel)
            elif op == "gamma":
                gamma = 2.2
                table = np.array([(i/255.0)**(1.0/gamma)*255 for i in range(256)]).astype("uint8")
                result = cv2.LUT(self.processed_cv, table)
            else:
                result = self.processed_cv.copy()
            
            self.processed_cv = result
            self.processed_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            self.update_displays()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error: {str(e)}")
        finally:
            self.processing = False
            self.show_progress(False)
        
    def add_noise(self, noise_type):
        if self.processed_cv is None:
            messagebox.showwarning("Warning", "⚠️ Load gambar terlebih dahulu!")
            return
        
        if self.processing:
            return
            
        self.show_progress(True, f"Adding {noise_type} noise...")
        self.processing = True
        self.root.after(100, lambda: self._do_add_noise(noise_type))
        
    def _do_add_noise(self, noise_type):
        try:
            intensity = self.noise_intensity.get()
            img = self.processed_cv.copy()
            
            if noise_type == "gaussian":
                noise = np.random.normal(0, intensity*255, img.shape)
                result = np.clip(img + noise, 0, 255).astype(np.uint8)
            elif noise_type == "salt_pepper":
                result = img.copy()
                prob = intensity/2
                salt = np.random.random(img.shape[:2]) < prob
                pepper = np.random.random(img.shape[:2]) < prob
                if len(img.shape) == 3:
                    result[salt] = [255,255,255]
                    result[pepper] = [0,0,0]
                else:
                    result[salt] = 255
                    result[pepper] = 0
            elif noise_type == "speckle":
                noise = np.random.normal(0, intensity, img.shape)
                result = np.clip(img + img*noise, 0, 255).astype(np.uint8)
            else:
                result = img.copy()
            
            self.processed_cv = result
            self.processed_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            self.update_displays()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error: {str(e)}")
        finally:
            self.processing = False
            self.show_progress(False)
        
    def apply_filter(self, filter_type):
        if self.processed_cv is None:
            messagebox.showwarning("Warning", "⚠️ Load gambar terlebih dahulu!")
            return
        
        if self.processing:
            return
            
        self.show_progress(True, f"Applying {filter_type} filter...")
        self.processing = True
        self.root.after(100, lambda: self._do_apply_filter(filter_type))
        
    def _do_apply_filter(self, filter_type):
        try:
            k = self.kernel.get()
            
            if filter_type == "average":
                result = cv2.blur(self.processed_cv, (k, k))
            elif filter_type == "gaussian":
                result = cv2.GaussianBlur(self.processed_cv, (k, k), 0)
            elif filter_type == "median":
                result = cv2.medianBlur(self.processed_cv, k)
            elif filter_type == "bilateral":
                result = cv2.bilateralFilter(self.processed_cv, 9, 75, 75)
            else:
                result = self.processed_cv.copy()
            
            self.processed_cv = result
            self.processed_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            self.update_displays()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error: {str(e)}")
        finally:
            self.processing = False
            self.show_progress(False)
        
    def analyze_fft_original(self):
        if self.original_cv is None:
            messagebox.showwarning("Warning", "⚠️ Load gambar terlebih dahulu!")
            self.show_fft_placeholder()
            return
        self.perform_fft(self.original_cv, "FFT Analysis - Original Image")
        
    def analyze_fft_processed(self):
        if self.processed_cv is None:
            messagebox.showwarning("Warning", "⚠️ Tidak ada gambar hasil!\nLoad gambar terlebih dahulu.")
            self.show_fft_placeholder()
            return
        self.perform_fft(self.processed_cv, "FFT Analysis - Processed Image")
        
    def perform_fft(self, img, title):
        try:
            # Convert to grayscale
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            
            # Perform FFT
            f = fft2(gray)
            fshift = fftshift(f)
            magnitude = np.log(np.abs(fshift) + 1)
            phase = np.angle(fshift)
            
            # Clear and plot
            self.fft_figure.clear()
            
            # Magnitude spectrum
            ax1 = self.fft_figure.add_subplot(1, 2, 1)
            im1 = ax1.imshow(magnitude, cmap='gray')
            ax1.set_title('Magnitude Spectrum', color='white', fontsize=12)
            ax1.axis('off')
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            
            # Phase spectrum
            ax2 = self.fft_figure.add_subplot(1, 2, 2)
            im2 = ax2.imshow(phase, cmap='gray')
            ax2.set_title('Phase Spectrum', color='white', fontsize=12)
            ax2.axis('off')
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            
            self.fft_figure.suptitle(title, fontsize=14, fontweight='bold', color='#4a90e2')
            self.fft_figure.tight_layout()
            self.fft_canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"FFT Error: {str(e)}")
        
    def apply_fft_filter(self):
        if self.processed_cv is None:
            messagebox.showwarning("Warning", "⚠️ Load gambar terlebih dahulu!")
            return
        
        if self.processing:
            return
            
        self.show_progress(True, "Applying FFT Low Pass Filter...")
        self.processing = True
        self.root.after(100, self._do_apply_fft_filter)
        
    def _do_apply_fft_filter(self):
        try:
            cutoff = self.cutoff.get()
            
            # Convert to grayscale
            if len(self.processed_cv.shape) == 3:
                gray = cv2.cvtColor(self.processed_cv, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.processed_cv.copy()
            
            # FFT
            f = fft2(gray)
            fshift = fftshift(f)
            
            # Create low pass filter mask
            rows, cols = gray.shape
            crow, ccol = rows//2, cols//2
            mask = np.zeros((rows, cols), np.uint8)
            r = int(cutoff * min(rows, cols)/2)
            y, x = np.ogrid[:rows, :cols]
            mask[(x-ccol)**2 + (y-crow)**2 <= r*r] = 1
            
            # Apply filter
            fshift_filtered = fshift * mask
            result = np.abs(ifft2(ifftshift(fshift_filtered)))
            result = np.clip(result, 0, 255).astype(np.uint8)
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            
            self.processed_cv = result
            self.processed_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            self.update_displays()
            messagebox.showinfo("Info", f"✅ Low Pass Filter applied!\nCutoff frequency: {cutoff:.2f}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Filter Error: {str(e)}")
        finally:
            self.processing = False
            self.show_progress(False)
        
    def show_histogram_original(self):
        if self.original_cv is None:
            messagebox.showwarning("Warning", "⚠️ Load gambar terlebih dahulu!")
            self.show_histogram_placeholder()
            return
        self.plot_histogram(self.original_cv, "Histogram - Original Image")
        
    def show_histogram_processed(self):
        if self.processed_cv is None:
            messagebox.showwarning("Warning", "⚠️ Tidak ada gambar hasil!\nLoad gambar terlebih dahulu.")
            self.show_histogram_placeholder()
            return
        self.plot_histogram(self.processed_cv, "Histogram - Processed Image")
        
    def show_both_histograms(self):
        if self.original_cv is None or self.processed_cv is None:
            messagebox.showwarning("Warning", "⚠️ Load gambar dan proses terlebih dahulu!")
            if self.original_cv is None:
                self.show_histogram_placeholder()
            return
        
        try:
            self.hist_figure.clear()
            
            # Original histogram
            ax1 = self.hist_figure.add_subplot(1, 2, 1)
            if len(self.original_cv.shape) == 3:
                colors = ['b', 'g', 'r']
                labels = ['Blue', 'Green', 'Red']
                for i, color in enumerate(colors):
                    hist = cv2.calcHist([self.original_cv], [i], None, [256], [0,256])
                    ax1.plot(hist, color=color, label=labels[i], linewidth=1.5)
                ax1.legend()
            else:
                hist = cv2.calcHist([self.original_cv], [0], None, [256], [0,256])
                ax1.plot(hist, color='white', linewidth=2)
            ax1.set_title('Original Image', color='white')
            ax1.set_xlabel('Pixel Intensity', color='#cccccc')
            ax1.set_ylabel('Frequency', color='#cccccc')
            ax1.grid(True, alpha=0.3)
            ax1.set_facecolor('#1e1e1e')
            ax1.tick_params(colors='#cccccc')
            
            # Processed histogram
            ax2 = self.hist_figure.add_subplot(1, 2, 2)
            if len(self.processed_cv.shape) == 3:
                colors = ['b', 'g', 'r']
                labels = ['Blue', 'Green', 'Red']
                for i, color in enumerate(colors):
                    hist = cv2.calcHist([self.processed_cv], [i], None, [256], [0,256])
                    ax2.plot(hist, color=color, label=labels[i], linewidth=1.5)
                ax2.legend()
            else:
                hist = cv2.calcHist([self.processed_cv], [0], None, [256], [0,256])
                ax2.plot(hist, color='white', linewidth=2)
            ax2.set_title('Processed Image', color='white')
            ax2.set_xlabel('Pixel Intensity', color='#cccccc')
            ax2.set_ylabel('Frequency', color='#cccccc')
            ax2.grid(True, alpha=0.3)
            ax2.set_facecolor('#1e1e1e')
            ax2.tick_params(colors='#cccccc')
            
            self.hist_figure.suptitle('Histogram Comparison', fontsize=14, fontweight='bold', color='#4a90e2')
            self.hist_figure.tight_layout()
            self.hist_canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Histogram Error: {str(e)}")
        
    def plot_histogram(self, img, title):
        try:
            self.hist_figure.clear()
            ax = self.hist_figure.add_subplot(111)
            
            if len(img.shape) == 3:
                colors = ['b', 'g', 'r']
                labels = ['Blue Channel', 'Green Channel', 'Red Channel']
                for i, color in enumerate(colors):
                    hist = cv2.calcHist([img], [i], None, [256], [0,256])
                    ax.plot(hist, color=color, label=labels[i], linewidth=2)
                ax.legend(loc='upper right')
            else:
                hist = cv2.calcHist([img], [0], None, [256], [0,256])
                ax.plot(hist, color='white', linewidth=2)
                ax.fill_between(range(256), hist.flatten(), alpha=0.3, color='gray')
            
            ax.set_title(title, fontsize=12, fontweight='bold', color='#4a90e2')
            ax.set_xlabel('Pixel Intensity', color='#cccccc')
            ax.set_ylabel('Frequency', color='#cccccc')
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 255])
            ax.set_facecolor('#1e1e1e')
            ax.tick_params(colors='#cccccc')
            
            # Add statistics
            mean_val = np.mean(img)
            std_val = np.std(img)
            stats_text = f'Mean: {mean_val:.2f}\nStd Dev: {std_val:.2f}'
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='#4a90e2', alpha=0.8),
                    color='white', fontsize=9)
            
            self.hist_figure.tight_layout()
            self.hist_canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Plot Error: {str(e)}")
        
    def reset_image(self):
        if self.original_cv is not None:
            self.processed_cv = self.original_cv.copy()
            self.processed_image = self.original_image.copy()
            self.update_displays()
            messagebox.showinfo("Info", "✅ Gambar telah direset ke original!")
            
    def save_image(self):
        if self.processed_image:
            path = filedialog.asksaveasfilename(defaultextension=".png", 
                                                filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")])
            if path:
                self.processed_image.save(path)
                messagebox.showinfo("Info", "✅ Gambar berhasil disimpan!")
                
    def update_displays(self):
        if self.original_image:
            img = self.original_image.resize((400,400), Image.Resampling.LANCZOS)
            self.orig_photo = ImageTk.PhotoImage(img)
            self.original_canvas.delete("all")
            self.original_canvas.create_image(200,200, image=self.orig_photo)
        if self.processed_image:
            img = self.processed_image.resize((400,400), Image.Resampling.LANCZOS)
            self.proc_photo = ImageTk.PhotoImage(img)
            self.result_canvas.delete("all")
            self.result_canvas.create_image(200,200, image=self.proc_photo)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()