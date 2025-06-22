import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ImageOperations import ImageOperations
from Descriptors import Descriptors


class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Processamento de Imagens")
        self.current_histogram_fig = None
        
        #estado da aplicação
        self.state = {
            'original_image': None,
            'processed_image': None,
            'current_image': None,
            'image_path': None,
            'is_gray': False
        }
        
        self.setup_ui()
    
    def setup_ui(self):
        self.setup_main_frame()
        self.setup_menu()
        self.setup_toolbar()
        self.setup_image_display()
        self.setup_status_bar()
    
    def setup_main_frame(self):
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def setup_menu(self):
        self.menu_bar = tk.Menu(self.root)
        
        #menu de arquivo
        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.file_menu.add_command(label="Abrir Imagem", command=self.load_image)
        self.file_menu.add_command(label="Salvar Imagem", command=self.save_image, state='disabled')
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Sair", command=self.root.quit)
        self.menu_bar.add_cascade(label="Arquivo", menu=self.file_menu)
        
        #menu de processamento
        self.setup_process_menu()
        
        #menu de extra
        self.setup_extra_menu()
        
        self.root.config(menu=self.menu_bar)
    
    def setup_process_menu(self):
        self.process_menu = tk.Menu(self.menu_bar, tearoff=0)
        
        #histograma
        self.process_menu.add_command(label="Histograma", command=self.show_histogram, state='disabled')
        
        #submenus
        self.setup_segmentation_menu()
        self.setup_morphology_menu()
        self.setup_intensity_menu()
        self.setup_lowpass_menu()
        self.setup_highpass_menu()
        self.setup_frequency_menu()
        
        self.menu_bar.add_cascade(label="Processamento", menu=self.process_menu)
    
    def setup_segmentation_menu(self):
        segmentation_menu = tk.Menu(self.process_menu, tearoff=0)
        segmentation_menu.add_command(label="Limiarização de Otsu", command=self.apply_otsu)
        self.process_menu.add_cascade(label="Segmentação", menu=segmentation_menu)
    
    def setup_morphology_menu(self):
        morph_menu = tk.Menu(self.process_menu, tearoff=0)
        morph_menu.add_command(label="Erosão", command=lambda: self.apply_morphology('erosion'))
        morph_menu.add_command(label="Dilatação", command=lambda: self.apply_morphology('dilation'))
        morph_menu.add_separator()
        morph_menu.add_command(label="Abertura", command=lambda: self.apply_morphology('opening'))
        morph_menu.add_command(label="Fechamento", command=lambda: self.apply_morphology('closing'))
        self.process_menu.add_cascade(label="Morfologia Matemática", menu=morph_menu)
    
    def setup_intensity_menu(self):
        intensity_menu = tk.Menu(self.process_menu, tearoff=0)
        intensity_menu.add_command(label="Alargamento de Contraste", command=self.contrast_stretching)
        intensity_menu.add_command(label="Equalização de Histograma", command=self.histogram_equalization)
        self.process_menu.add_cascade(label="Transformações de Intensidade", menu=intensity_menu)
    
    def setup_lowpass_menu(self):
        lowpass_menu = tk.Menu(self.process_menu, tearoff=0)
        lowpass_menu.add_command(label="Média", command=lambda: self.apply_filter('mean'))
        lowpass_menu.add_command(label="Mediana", command=lambda: self.apply_filter('median'))
        lowpass_menu.add_command(label="Gaussiano", command=lambda: self.apply_filter('gaussian'))
        lowpass_menu.add_command(label="Máximo", command=lambda: self.apply_filter('max'))
        lowpass_menu.add_command(label="Mínimo", command=lambda: self.apply_filter('min'))
        self.process_menu.add_cascade(label="Filtros Passa-Baixa", menu=lowpass_menu)
    
    def setup_highpass_menu(self):
        highpass_menu = tk.Menu(self.process_menu, tearoff=0)
        highpass_menu.add_command(label="Laplaciano", command=lambda: self.apply_filter('laplacian'))
        highpass_menu.add_command(label="Roberts", command=lambda: self.apply_filter('roberts'))
        highpass_menu.add_command(label="Prewitt", command=lambda: self.apply_filter('prewitt'))
        highpass_menu.add_command(label="Sobel", command=lambda: self.apply_filter('sobel'))
        self.process_menu.add_cascade(label="Filtros Passa-Alta", menu=highpass_menu)
    
    def setup_frequency_menu(self):
        freq_menu = tk.Menu(self.process_menu, tearoff=0)
        freq_menu.add_command(label="Filtro Passa-Baixa Ideal", command=lambda: self.frequency_filter('ideal_low'))
        freq_menu.add_command(label="Filtro Passa-Alta Ideal", command=lambda: self.frequency_filter('ideal_high'))
        freq_menu.add_command(label="Filtro Gaussiano Passa-Baixa", command=lambda: self.frequency_filter('gaussian_low'))
        freq_menu.add_command(label="Filtro Gaussiano Passa-Alta", command=lambda: self.frequency_filter('gaussian_high'))
        freq_menu.add_command(label="Espectro de Fourier", command=self.show_fourier_spectrum)
        self.process_menu.add_cascade(label="Domínio da Frequência", menu=freq_menu)
    
    def setup_extra_menu(self):
        self.extra_menu = tk.Menu(self.menu_bar, tearoff=0)
        descriptors_menu = tk.Menu(self.extra_menu, tearoff=0)
        
        descriptors_menu.add_command(label="Histograma de Cores (Intensidade)", command=self.show_intensity_histogram)
        descriptors_menu.add_command(label="Descritores de Textura (Haralick)", command=self.calculate_haralick)
        descriptors_menu.add_command(label="Descritores de Forma (Moments)", command=self.calculate_shape_moments)
        descriptors_menu.add_command(label="Descritores de Cor (Intensidade Média)", command=self.calculate_intensity_stats)
        
        self.extra_menu.add_cascade(label="Descritores de Imagem", menu=descriptors_menu)
        self.menu_bar.add_cascade(label="Extra", menu=self.extra_menu)
    
    def setup_toolbar(self):
        self.toolbar = tk.Frame(self.root, bd=1, relief=tk.RAISED)
        self.toolbar.pack(side=tk.TOP, fill=tk.X)
        
        self.reset_button = tk.Button(self.toolbar, text="Resetar Imagem", 
                                    command=self.reset_image, state='disabled')
        self.reset_button.pack(side=tk.RIGHT, padx=5, pady=2)
    
    def setup_image_display(self):
        self.image_frame = tk.Frame(self.main_frame)
        self.image_frame.pack(pady=10)
        
        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack()
    
    def setup_status_bar(self):
        self.status_bar = tk.Label(self.root, text="Pronto :)", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(fill=tk.X)
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Selecione uma imagem",
            filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("Todos os arquivos", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            self.update_status(f"Carregando imagem: {os.path.basename(file_path)}...")
            image = Image.open(file_path)
            
            if image.mode != 'L':
                image = ImageOps.grayscale(image)
                self.state['is_gray'] = True
            else:
                self.state['is_gray'] = True
            
            self.state.update({
                'original_image': image.copy(),
                'processed_image': None,
                'current_image': image.copy(),
                'image_path': file_path
            })
            
            self.display_image(image)
            self.enable_image_operations()
            self.update_status(f"Imagem carregada: {os.path.basename(file_path)}")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Não foi possível carregar a imagem:\n{str(e)}")
            self.update_status("Erro ao carregar imagem")
    
    def enable_image_operations(self):
        self.file_menu.entryconfig("Salvar Imagem", state='normal')
        self.process_menu.entryconfig("Histograma", state='normal')
        self.reset_button.config(state='normal')
        self.extra_menu.entryconfig(0, state='normal')
    
    def save_image(self):
        if not self.state['current_image']:
            messagebox.showwarning("Aviso", "Nenhuma imagem para salvar")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Salvar imagem",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp"), ("TIFF", "*.tif"), ("Todos os arquivos", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            self.update_status(f"Salvando imagem: {os.path.basename(file_path)}...")
            self.state['current_image'].save(file_path)
            self.update_status(f"Imagem salva: {os.path.basename(file_path)}")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Não foi possível salvar a imagem:\n{str(e)}")
            self.update_status("Erro ao salvar imagem")
    
    def display_image(self, image):
        window_width = self.root.winfo_width()
        window_height = self.root.winfo_height()
        
        #calcula espaço disponível
        status_height = self.status_bar.winfo_height()
        toolbar_height = self.toolbar.winfo_height()
        menu_height = 30
        available_height = window_height - status_height - toolbar_height - menu_height - 40
        
        if window_width < 100 or available_height < 100:
            window_width = 600
            available_height = 400
        
        #calcula ratio de redimensionamento
        img_width, img_height = image.size
        ratio = min((window_width - 40) / img_width, available_height / img_height)
        ratio = min(ratio, 1.0)
        
        new_size = (int(img_width * ratio), int(img_height * ratio))
        resized_image = image.resize(new_size, Image.LANCZOS)
        
        self.tk_image = ImageTk.PhotoImage(resized_image)
        self.image_label.config(image=self.tk_image)
        self.image_label.image = self.tk_image
        self.image_frame.config(width=new_size[0], height=new_size[1])
    
    def reset_image(self):
        if self.state['original_image']:
            self.state['current_image'] = self.state['original_image'].copy()
            self.state['processed_image'] = None
            self.display_image(self.state['current_image'])
            self.update_status("Imagem resetada para o original")
    
    def update_status(self, message):
        self.status_bar.config(text=message)
        self.root.update_idletasks()
    
    def apply_otsu(self):
        if not self.state['current_image']:
            messagebox.showwarning("Aviso", "Nenhuma imagem carregada")
            return
            
        try:
            self.update_status("Aplicando limiarização de Otsu...")
            processed_img, threshold = ImageOperations.apply_otsu(self.state['current_image'])
            self.update_image_state(processed_img)
            
            self.update_status(f"Limiarização de Otsu aplicada (Threshold: {threshold:.2f})")
            self.show_histogram(threshold=threshold)
            
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao aplicar Otsu:\n{str(e)}")
            self.update_status("Erro ao aplicar limiarização")
    
    def update_image_state(self, processed_img):
        self.state['processed_image'] = processed_img
        self.state['current_image'] = processed_img
        self.display_image(processed_img)
    
    def contrast_stretching(self):
        if not self.state['current_image']:
            messagebox.showwarning("Aviso", "Nenhuma imagem carregada")
            return
            
        try:
            self.update_status("Aplicando alargamento de contraste...")
            processed_img = ImageOperations.contrast_stretching(self.state['current_image'])
            self.update_image_state(processed_img)
            self.update_status("Alargamento de contraste aplicado")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao aplicar alargamento de contraste:\n{str(e)}")
            self.update_status("Erro ao processar imagem")
    
    def histogram_equalization(self):
        if not self.state['current_image']:
            messagebox.showwarning("Aviso", "Nenhuma imagem carregada")
            return
            
        try:
            self.update_status("Aplicando equalização de histograma...")
            processed_img = ImageOperations.histogram_equalization(self.state['current_image'])
            self.update_image_state(processed_img)
            self.update_status("Equalização de histograma aplicada")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao aplicar equalização de histograma:\n{str(e)}")
            self.update_status("Erro ao processar imagem")
    
    def apply_filter(self, filter_type):
        if not self.state['current_image']:
            messagebox.showwarning("Aviso", "Nenhuma imagem carregada")
            return
            
        try:
            self.update_status(f"Aplicando filtro {filter_type}...")
            processed_img = ImageOperations.apply_filter(self.state['current_image'], filter_type)
            self.update_image_state(processed_img)
            self.update_status(f"Filtro {filter_type} aplicado com sucesso")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao aplicar filtro:\n{str(e)}")
            self.update_status("Erro ao processar imagem")
    
    def frequency_filter(self, filter_type):
        if not self.state['current_image']:
            messagebox.showwarning("Aviso", "Nenhuma imagem carregada")
            return

        try:
            self.update_status(f"Aplicando filtro {filter_type}...")
            processed_img = ImageOperations.frequency_filter(self.state['current_image'], filter_type)
            self.update_image_state(processed_img)
            self.update_status(f"Filtro {filter_type} aplicado com sucesso")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao aplicar filtro de frequência:\n{str(e)}")
            self.update_status("Erro ao processar imagem")
    
    def apply_morphology(self, operation):
        if not self.state['current_image']:
            messagebox.showwarning("Aviso", "Nenhuma imagem carregada")
            return
            
        try:
            self.update_status(f"Aplicando {operation}...")
            processed_img = ImageOperations.apply_morphology(self.state['current_image'], operation)
            self.update_image_state(processed_img)
            self.update_status(f"{operation} aplicada com sucesso")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao aplicar {operation}:\n{str(e)}")
            self.update_status(f"Erro ao aplicar {operation}")

    def show_histogram(self, threshold=None):
        if not self.state['current_image']:
            messagebox.showwarning("Aviso", "Nenhuma imagem carregada para calcular o histograma")
            return
        
        try:
            img_array = np.array(self.state['current_image'])
            
            #verifica se é imagem binária (pós-Otsu)
            unique_vals = np.unique(img_array)
            is_binary = len(unique_vals) <= 2 and (0 in unique_vals and 255 in unique_vals)
            
            if is_binary:
                #histograma especial para imagens binárias
                hist = np.zeros(256)
                count_0 = np.sum(img_array == 0)
                count_255 = np.sum(img_array == 255)
                hist[0] = count_0
                hist[255] = count_255
                
                #cria figura com ajustes para binário
                fig = plt.Figure(figsize=(6, 4), dpi=100)
                ax = fig.add_subplot(111)
                
                #plota apenas as barras relevantes
                ax.bar([0, 255], [count_0, count_255], width=10, color='gray')
                ax.set_title("Histograma Binário (Pós-Otsu)")
                ax.set_xlabel("Valores de Pixel")
                ax.set_ylabel("Frequência")
                ax.set_xlim(-10, 265)  #espaço para visualização
                ax.set_xticks([0, 255])  #mostra apenas 0 e 255 no eixo X
                
            else:
                #histograma normal para imagens não-binárias
                hist = ImageOperations.calculate_histogram(self.state['current_image'])
                
                fig = plt.Figure(figsize=(6, 4), dpi=100)
                ax = fig.add_subplot(111)
                ax.bar(range(256), hist, width=1, color='gray')
                ax.set_title("Histograma de Tons de Cinza")
                ax.set_xlabel("Intensidade")
                ax.set_ylabel("Frequência")
                ax.set_xlim(0, 255)
            
            #adiciona linha do threshold se fornecido
            if threshold is not None:
                ax.axvline(x=threshold, color='r', linestyle='dashed', linewidth=2)
                ax.text(threshold+5, ax.get_ylim()[1]*0.9, 
                    f'Threshold: {threshold:.1f}', color='red')
            
            #configuração da janela
            hist_window = tk.Toplevel(self.root)
            hist_window.title("Histograma")
            
            canvas = FigureCanvasTkAgg(fig, master=hist_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            #botão de salvar
            btn_frame = tk.Frame(hist_window)
            btn_frame.pack(fill=tk.X, padx=5, pady=5)
            tk.Button(btn_frame, text="Fechar", command=hist_window.destroy).pack(side=tk.RIGHT)
            tk.Button(btn_frame, text="Salvar", 
                    command=lambda: self.save_histogram(fig, hist_window)).pack(side=tk.RIGHT, padx=5)
            
            #mantém referências
            self.current_histogram_fig = fig
            hist_window._canvas = canvas
            
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao exibir histograma:\n{str(e)}")

    def save_histogram(self, fig, parent_window):
        if fig is None:
            messagebox.showwarning("Aviso", "Nenhum histograma para salvar", parent=parent_window)
            return
            
        file_path = filedialog.asksaveasfilename(
            parent=parent_window,
            title="Salvar Histograma como Imagem",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("PDF", "*.pdf"), ("Todos os arquivos", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            fig.savefig(file_path, bbox_inches='tight', dpi=100)
            messagebox.showinfo("Sucesso", f"Histograma salvo em:\n{file_path}", parent=parent_window)
        except Exception as e:
            messagebox.showerror("Erro", f"Não foi possível salvar o histograma:\n{str(e)}", parent=parent_window)
    
    def show_fourier_spectrum(self):
        if not self.state['current_image']:
            messagebox.showwarning("Aviso", "Nenhuma imagem carregada")
            return

        try:
            self.update_status("Calculando espectro de Fourier...")
            spectrum_img = ImageOperations.calculate_fourier_spectrum(self.state['current_image'])
            
            spectrum_window = tk.Toplevel(self.root)
            spectrum_window.title("Espectro de Fourier")
            spectrum_window.geometry("600x500")
            
            control_frame = tk.Frame(spectrum_window)
            control_frame.pack(fill=tk.X, padx=5, pady=5)
            
            save_button = tk.Button(control_frame, text="Salvar Espectro", 
                                command=lambda: self.save_spectrum(spectrum_img, spectrum_window))
            save_button.pack(side=tk.RIGHT, padx=5)
            
            tk_spectrum = ImageTk.PhotoImage(spectrum_img)
            label = tk.Label(spectrum_window, image=tk_spectrum)
            label.image = tk_spectrum
            label.pack(fill=tk.BOTH, expand=True)
            
            self.current_spectrum_img = spectrum_img
            self.update_status("Espectro de Fourier calculado")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao calcular espectro:\n{str(e)}")
            self.update_status("Erro ao processar espectro")
    
    def save_spectrum(self, spectrum_img, parent_window):
        if not hasattr(self, 'current_spectrum_img') or self.current_spectrum_img is None:
            messagebox.showwarning("Aviso", "Nenhum espectro para salvar", parent=parent_window)
            return
            
        file_path = filedialog.asksaveasfilename(
            parent=parent_window,
            title="Salvar Espectro como Imagem",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("TIFF", "*.tif"), ("Todos os arquivos", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            spectrum_img.save(file_path)
            messagebox.showinfo("Sucesso", f"Espectro salvo em:\n{file_path}", parent=parent_window)
        except Exception as e:
            messagebox.showerror("Erro", f"Não foi possível salvar o espectro:\n{str(e)}", parent=parent_window)
    
    def show_intensity_histogram(self):
        if not self.state['current_image']:
            messagebox.showwarning("Aviso", "Nenhuma imagem carregada")
            return
            
        try:
            hist = ImageOperations.calculate_histogram(self.state['current_image'])
            stats = Descriptors.calculate_intensity_stats(self.state['current_image'])
            
            fig = plt.Figure(figsize=(8, 5), dpi=100)
            ax = fig.add_subplot(111)
            ax.bar(range(256), hist, width=1, color='gray')
            ax.set_title("Histograma de Intensidade (Tons de Cinza)")
            ax.set_xlabel("Valor de Intensidade")
            ax.set_ylabel("Frequência")
            
            stats_text = f"Média: {stats['mean']:.2f}\nDesvio Padrão: {stats['std']:.2f}\nMediana: {stats['median']:.2f}"
            ax.text(0.7, 0.9, stats_text, transform=ax.transAxes, 
                   bbox=dict(facecolor='white', alpha=0.8))
            
            stats_window = tk.Toplevel(self.root)
            stats_window.title("Histograma e Estatísticas de Intensidade")
            
            canvas = FigureCanvasTkAgg(fig, master=stats_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            save_button = tk.Button(stats_window, text="Salvar Gráfico", 
                                  command=lambda: self.save_histogram(fig, stats_window))
            save_button.pack(pady=5)
            
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao calcular histograma:\n{str(e)}")
    
    def calculate_haralick(self):
        if not self.state['current_image']:
            messagebox.showwarning("Aviso", "Nenhuma imagem carregada")
            return
            
        try:
            features = Descriptors.calculate_haralick_features(self.state['current_image'])
            
            result_window = tk.Toplevel(self.root)
            result_window.title("Descritores de Textura - Haralick")
            result_frame = tk.Frame(result_window, padx=10, pady=10)
            result_frame.pack(fill=tk.BOTH, expand=True)
            
            tk.Label(result_frame, text="Características de Textura (Haralick)", 
                   font=('Arial', 12, 'bold')).pack(pady=5)
            
            metrics = [
                ("Contraste:", features['contrast']),
                ("Dissimilaridade:", features['dissimilarity']),
                ("Homogeneidade:", features['homogeneity']),
                ("Energia:", features['energy']),
                ("Correlação:", features['correlation']),
                ("Momento Angular Segundo (ASM):", features['asm'])
            ]
            
            for name, value in metrics:
                frame = tk.Frame(result_frame)
                frame.pack(fill=tk.X, pady=2)
                tk.Label(frame, text=name, width=25, anchor='w').pack(side=tk.LEFT)
                tk.Label(frame, text=f"{value:.4f}").pack(side=tk.LEFT)
            
            tk.Button(result_window, text="Fechar", command=result_window.destroy).pack(pady=10)
            
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao calcular características de Haralick:\n{str(e)}")
    
    def calculate_shape_moments(self):
        if not self.state['current_image']:
            messagebox.showwarning("Aviso", "Nenhuma imagem carregada")
            return
            
        try:
            moments = Descriptors.calculate_shape_moments(self.state['current_image'])
            
            result_window = tk.Toplevel(self.root)
            result_window.title("Descritores de Forma - Momentos")
            result_frame = tk.Frame(result_window, padx=10, pady=10)
            result_frame.pack(fill=tk.BOTH, expand=True)
            
            tk.Label(result_frame, text="Momentos de Forma", 
                   font=('Arial', 12, 'bold')).pack(pady=5)
            
            # Momentos espaciais
            tk.Label(result_frame, text="Momentos Espaciais:", 
                   font=('Arial', 10, 'underline')).pack(anchor='w', pady=5)
            
            spatial_moments = [
                ("m00:", moments['spatial_moments']['m00']),
                ("m10:", moments['spatial_moments']['m10']),
                ("m01:", moments['spatial_moments']['m01']),
                ("m20:", moments['spatial_moments']['m20']),
                ("m11:", moments['spatial_moments']['m11']),
                ("m02:", moments['spatial_moments']['m02'])
            ]
            
            for name, value in spatial_moments:
                frame = tk.Frame(result_frame)
                frame.pack(fill=tk.X, pady=2)
                tk.Label(frame, text=name, width=10, anchor='w').pack(side=tk.LEFT)
                tk.Label(frame, text=f"{value:.4f}").pack(side=tk.LEFT)
            
            #momentos centrais
            tk.Label(result_frame, text="Momentos Centrais:", 
                   font=('Arial', 10, 'underline')).pack(anchor='w', pady=5)
            
            central_moments = [
                ("mu20:", moments['central_moments']['mu20']),
                ("mu11:", moments['central_moments']['mu11']),
                ("mu02:", moments['central_moments']['mu02']),
                ("mu30:", moments['central_moments']['mu30']),
                ("mu21:", moments['central_moments']['mu21']),
                ("mu12:", moments['central_moments']['mu12']),
                ("mu03:", moments['central_moments']['mu03'])
            ]
            
            for name, value in central_moments:
                frame = tk.Frame(result_frame)
                frame.pack(fill=tk.X, pady=2)
                tk.Label(frame, text=name, width=10, anchor='w').pack(side=tk.LEFT)
                tk.Label(frame, text=f"{value:.4f}").pack(side=tk.LEFT)
            
            #momentos de Hu
            tk.Label(result_frame, text="Momentos de Hu (Invariantes):", 
                   font=('Arial', 10, 'underline')).pack(anchor='w', pady=5)
            
            for i in range(7):
                frame = tk.Frame(result_frame)
                frame.pack(fill=tk.X, pady=2)
                tk.Label(frame, text=f"Hu{i+1}:", width=10, anchor='w').pack(side=tk.LEFT)
                tk.Label(frame, text=f"{moments['hu_moments'][i]:.4e}").pack(side=tk.LEFT)
            
            tk.Button(result_window, text="Fechar", command=result_window.destroy).pack(pady=10)
            
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao calcular momentos de forma:\n{str(e)}")
    
    def calculate_intensity_stats(self):
        if not self.state['current_image']:
            messagebox.showwarning("Aviso", "Nenhuma imagem carregada")
            return
            
        try:
            stats = Descriptors.calculate_intensity_stats(self.state['current_image'])
            
            result_window = tk.Toplevel(self.root)
            result_window.title("Estatísticas de Intensidade (Cor)")
            result_frame = tk.Frame(result_window, padx=10, pady=10)
            result_frame.pack(fill=tk.BOTH, expand=True)
            
            tk.Label(result_frame, text="Estatísticas de Intensidade (Tons de Cinza)", 
                   font=('Arial', 12, 'bold')).pack(pady=5)
            
            metrics = [
                ("Média:", stats['mean']),
                ("Desvio Padrão:", stats['std']),
                ("Mediana:", stats['median']),
                ("Valor Mínimo:", stats['min']),
                ("Valor Máximo:", stats['max']),
                ("Energia:", stats['energy']),
                ("Entropia:", stats['entropy'])
            ]
            
            for name, value in metrics:
                frame = tk.Frame(result_frame)
                frame.pack(fill=tk.X, pady=2)
                tk.Label(frame, text=name, width=15, anchor='w').pack(side=tk.LEFT)
                tk.Label(frame, text=f"{value:.4f}").pack(side=tk.LEFT)
            
            tk.Button(result_window, text="Fechar", command=result_window.destroy).pack(pady=10)
            
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao calcular estatísticas:\n{str(e)}")
    
    def on_resize(self, event):
        if self.state['current_image']:
            self.display_image(self.state['current_image'])

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("800x600")
    app = ImageProcessingApp(root)
    root.bind('<Configure>', app.on_resize)
    root.mainloop()