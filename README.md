# Sistema de Processamento de Imagens - SIN392

## Descrição
Trabalho para a disciplina SIN392 - Processamento Digital de Imagens. Este projeto implementa um sistema de processamento de imagens com interface gráfica, contendo operações de processamento digital. 

*Link para o vídeo da apresentação do trabalho:* https://youtu.be/R2EYflkA9Qc

O sistema permite:

- Carregar e salvar imagens em diversos formatos
- Aplicar transformações de intensidade
- Executar filtragens espaciais e no domínio da frequência
- Realizar operações morfológicas
- Extrair descritores de características
- Visualizar resultados através de histogramas e espectros

## Principais Funcionalidades

### Processamento Básico
- Carregamento de imagens em níveis de cinza
- Visualização interativa com redimensionamento automático
- Salvamento de imagens processadas

### Transformações de Intensidade
- Visualização do histograma da imagem
- Alargamento de contraste adaptativo
- Equalização de histograma
- Limiarização automática (Otsu)
- Visualização de histogramas com marcação de threshold

### Filtragem Espacial
**Filtros passa-baixa:**
- Média
- Mediana
- Gaussiano
- Máximo
- Mínimo

**Filtros passa-alta:**
- Laplaciano
- Roberts
- Prewitt
- Sobel

### Domínio da Frequência
- Filtragem no domínio da frequência (passa-baixa/passa-alta)
- Visualização do espectro de Fourier
- Filtros ideais e gaussianos

### Morfologia Matemática
- Erosão
- Dilatação
- Abertura
- Fechamento

### Descritores de Imagem
- Estatísticas de intensidade (média, desvio padrão, etc.)
- Características de Haralick (textura)
- Momentos invariantes (forma)

## Estrutura do Código

### 1. Módulos Principais
- `ImageOperations.py`: Contém os algoritmos de transformação de imagens
- `Descriptors.py`: Contém os algoritmos de extração de características
- `ImageProcessingApp.py`: Interface gráfica baseada em Tkinter e operações de processamento

### 2. Organização da Interface
- Menu principal com todas as operações categorizadas
- Barra de ferramentas para ações rápidas
- Área de visualização de imagens
- Barra de status informativa

### 3. Fluxo de Processamento
1. Carregamento da imagem original
2. Aplicação das operações selecionadas
3. Visualização dos resultados
4. Opção de salvar ou resetar

## Pré-requisitos
- Python 3.6+
- Bibliotecas necessárias:
  ```bash
  pip install opencv-python pillow numpy scipy scikit-image matplotlib
  ```

## Como Executar
1. Clone o repositório:
   ```bash
   git clone https://github.com/grazistefane/TrabalhoPDI_SIN392.git
   cd TrabalhoPDI_SIN392
   ```

2. Execute o programa principal:
   ```bash
   python ImageProcessingApp.py
   ```

## Resultados Esperados
- Interface gráfica funcional e intuitiva
- Capacidade de aplicar diversas técnicas de PDI
- Visualização imediata dos resultados
- Extração e exibição de descritores de imagem


## Integrante do Projeto
- Grazielle Stefane Cruz - 7021. email: grazielle.cruz@ufv.br
