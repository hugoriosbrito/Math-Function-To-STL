
# Gerador 3D por Função Matemática

Este projeto é uma aplicação interativa desenvolvida com [Streamlit](https://streamlit.io/) que permite criar modelos 3D sólidos no formato `.stl` a partir de funções matemáticas. Os modelos gerados podem ser visualizados e baixados para impressão 3D.

## Funcionalidades

- **Entrada de Funções Matemáticas**: Insira uma função personalizada ou escolha uma das funções pré-definidas.
- **Visualização 3D**: Visualize o modelo gerado diretamente na interface.
- **Download de Arquivos STL**: Baixe o modelo gerado no formato `.stl` para impressão 3D.
- **Parâmetros Personalizáveis**: Ajuste resolução, amplitude e tamanho da base para personalizar o modelo.

## Tecnologias Utilizadas

- **Python**: Linguagem principal do projeto.
- **Streamlit**: Framework para criação de interfaces interativas.
- **NumPy**: Biblioteca para cálculos matemáticos e manipulação de arrays.
- **Matplotlib**: Para visualização 3D dos modelos.
- **NumPy-STL**: Para geração e manipulação de arquivos STL.

## Como Funciona

1. **Entrada da Função**:
   - Escolha uma função pré-definida ou insira uma função personalizada no formato `z = f(x, y)`.
   - Funções permitidas incluem `sin`, `cos`, `sqrt`, `exp`, entre outras.

2. **Configuração de Parâmetros**:
   - **Resolução**: Define o número de pontos na malha (quanto maior, mais detalhado).
   - **Amplitude**: Multiplicador para a altura (eixo Z).
   - **Tamanho da Base**: Define o intervalo para os eixos X e Y.

3. **Geração do Modelo**:
   - A função é avaliada para criar uma malha 3D.
   - O modelo é visualizado em um gráfico 3D interativo.

4. **Download do Arquivo STL**:
   - O modelo gerado pode ser baixado no formato `.stl` para impressão 3D.

## Como Executar

### Pré-requisitos

- Python 3.8 ou superior.
- Instale as dependências listadas no arquivo `requirements.txt`:
  ```bash
  pip install -r requirements.txt
  ```

### Executando o Projeto

1. Clone este repositório:
   ```bash
   git clone https://github.com/seu-usuario/functo3D.git
   cd functo3D
   ```

2. Execute o aplicativo Streamlit:
   ```bash
   streamlit run main.py
   ```

3. Acesse o aplicativo no navegador em `http://localhost:8501`.

## Estrutura do Projeto

```
functo3D/
├── main.py          # Código principal do aplicativo
├── README.md        # Documentação do projeto
├── requirements.txt # Dependências do projeto
```

## Exemplos de Funções

- **Ondas Circulares**: `sin(sqrt(x**2 + y**2))`
- **Sela**: `0.2 * (x**2 - y**2)`
- **Vulcão Simples**: `exp(-(x**2 + y**2) / (0.5*size)) * size`
- **Grades Cruzadas**: `cos(x) * sin(y)`
- **Paraboloide**: `0.1 * (x**2 + y**2)`

## Contribuindo

Contribuições são bem-vindas! Siga os passos abaixo para contribuir:

1. Faça um fork do repositório.
2. Crie uma branch para sua feature ou correção:
   ```bash
   git checkout -b minha-feature
   ```
3. Faça commit das suas alterações:
   ```bash
   git commit -m "Minha nova feature"
   ```
4. Envie para o repositório remoto:
   ```bash
   git push origin minha-feature
   ```
5. Abra um Pull Request.

## Contato

Para dúvidas ou sugestões, entre em contato:

- **Autor**: Hugo Rios Brito
- **Email**: hugoba532@gmail.com
