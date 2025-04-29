import streamlit as st
import numpy as np
import tempfile
from stl import mesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import traceback # Para logging de erros detalhado, se necessário

# --- Configurações Iniciais ---
st.set_page_config(
    layout="wide",
    page_title="Gerador 3D por Função",
    page_icon="🧊"
)

# --- Funções de Geração da Malha (Cacheada) ---

@st.cache_data(max_entries=50) # Cacheia os últimos 50 resultados únicos
def calcular_malha(res, ampl, size, func_str):
    """
    Calcula os vértices e faces da malha 3D com base nos parâmetros.
    Esta função é cacheada para performance.

    Args:
        res (int): Resolução da malha.
        ampl (float): Amplitude (multiplicador de altura Z).
        size (float): Tamanho da base (intervalo [-size, size] para x e y).
        func_str (str): String da função matemática z = f(x, y).

    Returns:
        tuple: Contendo (x, y, z, all_vertices, faces, x_vals, y_vals)
               Retorna None em caso de erro de cálculo.
        str: Mensagem de erro, se houver. None se sucesso.
    """
    if res < 4: # Validação mínima para evitar malhas degeneradas
        return None, "Erro: A resolução deve ser pelo menos 4."

    try:
        # Prepara variáveis para avaliação segura
        x_vals = np.linspace(-size, size, res)
        y_vals = np.linspace(-size, size, res)
        x, y = np.meshgrid(x_vals, y_vals)

        # --- Avaliação Segura da Função ---
        # Ambiente limitado para segurança. Funções permitidas:
        safe_dict = {
            "x": x, "y": y, "np": np, # Permitir np pode ser um risco se não confiar nos usuários
            "sin": np.sin, "cos": np.cos, "tan": np.tan,
            "sqrt": np.sqrt, "exp": np.exp, "log": np.log, "log10": np.log10,
            "abs": np.abs, "pi": np.pi, "size": size,
            "cosh": np.cosh, "sinh": np.sinh, "tanh": np.tanh,
            "power": np.power, # Ex: power(x, 2) para x**2
        }
       
        z_raw = ampl * eval(func_str, {"__builtins__": None}, safe_dict)

        # Validação do resultado de Z (deve ser um array numérico)
        if not isinstance(z_raw, np.ndarray) or not np.issubdtype(z_raw.dtype, np.number):
             raise ValueError("A função não retornou um resultado numérico válido.")

        # Garante que a menor altura seja zero (a base toca o plano z=0)
        min_z = np.min(z_raw)
        max_z = np.max(z_raw)
        if np.isclose(min_z, max_z):
             z = np.zeros_like(z_raw) if np.isclose(min_z, 0) else z_raw - min_z
        else:
            z = z_raw - min_z
        # --- Fim da Avaliação Segura ---

        # --- Criação de Vértices e Faces (Lógica inalterada) ---
        x_flat, y_flat, z_flat = x.flatten(), y.flatten(), z.flatten()
        num_vertices_per_layer = res * res

        vertices_top = np.column_stack((x_flat, y_flat, z_flat))
        base_z = np.zeros_like(z_flat)
        vertices_base = np.column_stack((x_flat, y_flat, base_z))
        all_vertices = np.vstack((vertices_top, vertices_base))

        faces = []
        # 1. Faces Superiores
        for i in range(res - 1):
            for j in range(res - 1):
                idx = i * res + j
                v1, v2, v3, v4 = idx, idx + 1, idx + res, idx + res + 1
                faces.append([v1, v3, v2])
                faces.append([v2, v3, v4])

        # 2. Faces Inferiores
        offset = num_vertices_per_layer
        for i in range(res - 1):
            for j in range(res - 1):
                idx = i * res + j
                v1b, v2b, v3b, v4b = idx + offset, idx + 1 + offset, idx + res + offset, idx + res + 1 + offset
                faces.append([v1b, v2b, v3b])
                faces.append([v2b, v4b, v3b])

        # 3. Faces Laterais
        for i in range(res - 1):
            # Direita
            idx_t1, idx_t2 = i*res + res-1, (i+1)*res + res-1
            idx_b1, idx_b2 = idx_t1 + offset, idx_t2 + offset
            faces.append([idx_t1, idx_b1, idx_t2]); faces.append([idx_b1, idx_b2, idx_t2])
            # Esquerda
            idx_t1, idx_t2 = i*res, (i+1)*res
            idx_b1, idx_b2 = idx_t1 + offset, idx_t2 + offset
            faces.append([idx_t1, idx_t2, idx_b1]); faces.append([idx_b1, idx_t2, idx_b2])
        for j in range(res - 1):
            # Superior
            idx_t1, idx_t2 = (res-1)*res + j, (res-1)*res + j+1
            idx_b1, idx_b2 = idx_t1 + offset, idx_t2 + offset
            faces.append([idx_t1, idx_t2, idx_b1]); faces.append([idx_b1, idx_t2, idx_b2])
            # Inferior
            idx_t1, idx_t2 = j, j+1
            idx_b1, idx_b2 = idx_t1 + offset, idx_t2 + offset
            faces.append([idx_t1, idx_b1, idx_t2]); faces.append([idx_b1, idx_b2, idx_t2])
        # --- Fim da Criação de Vértices e Faces ---

        return (x, y, z, all_vertices, faces, x_vals, y_vals), None # Sucesso

    except Exception as e:
        # Log detalhado do erro no console do Streamlit (para o desenvolvedor)
        print(f"Erro detalhado em calcular_malha para func='{func_str}':")
        print(traceback.format_exc())
        # Mensagem de erro para o usuário
        user_error_message = f"Erro ao processar a função '{func_str}': {e}. Verifique a sintaxe e as funções permitidas."
        return None, user_error_message


# --- Função para Salvar STL (Não Cacheada) ---
def salvar_stl(all_vertices, faces, file_prefix="modelo_stl"):
    """
    Cria um objeto mesh e salva em um arquivo STL temporário.

    Args:
        all_vertices (np.ndarray): Array de vértices.
        faces (list): Lista de faces (listas de índices de vértices).
        file_prefix (str): Prefixo para o nome do arquivo temporário.

    Returns:
        str: Caminho para o arquivo STL temporário salvo. None se erro.
        str: Mensagem de erro, se houver. None se sucesso.
    """
    try:
        num_faces = len(faces)
        if num_faces == 0:
            return None, "Erro: Nenhum face gerada para criar o STL."

        # Criação da malha STL
        solid_mesh = mesh.Mesh(np.zeros(num_faces, dtype=mesh.Mesh.dtype))
        for i, face in enumerate(faces):
            for j in range(3):
                solid_mesh.vectors[i][j] = all_vertices[face[j], :]

        # Salva em arquivo temporário seguro
        with tempfile.NamedTemporaryFile(delete=False, suffix='.stl', prefix=file_prefix + "_") as temp_file:
            solid_mesh.save(temp_file.name)
            temp_file_path = temp_file.name

        return temp_file_path, None # Sucesso

    except Exception as e:
        print(f"Erro detalhado ao salvar STL:")
        print(traceback.format_exc())
        user_error_message = f"Erro ao criar ou salvar o arquivo STL: {e}"
        return None, user_error_message

# === Interface Streamlit ===

st.title("🧊 Gerador de Modelo 3D (STL) por Função Matemática")
st.markdown("""
Crie modelos 3D sólidos e prontos para impressão FDM (.stl) inserindo uma função matemática $z = f(x, y)$ ou escolhendo uma da lista.
**Nota:** A visualização é uma aproximação. Use o visualizador externo para o modelo exato.
""")
st.markdown("---")

# --- Dicionário de Funções Pré-definidas ---
predefined_functions = {
    "Ondas Circulares": "sin(sqrt(x**2 + y**2))",
    "Sela": "0.2 * (x**2 - y**2)",
    "Vulcão Simples": "exp(-(x**2 + y**2) / (0.5*size)) * size",
    "Grades Cruzadas": "cos(x) * sin(y)",
    "Picos Senoidais": "sin(pi*x/size) * sin(pi*y/size) * size/2",
    "Paraboloide": "0.1 * (x**2 + y**2)",
    "Ripples": "sin(10 * sqrt(x**2 + y**2)) / 10",
    "Função 'Peaks' MATLAB": "3*(1-x)**2*exp(-(x**2) - (y+1)**2) - 10*(x/5 - x**3 - y**5)*exp(-x**2-y**2) - 1/3*exp(-(x+1)**2 - y**2)",
    "Terreno Fractal Simples": "0.3*sin(x/2)*cos(y/3) + 0.2*sin(x*2)*cos(y*1.5) + 0.1*sin(x*5)*cos(y*4)"
}

col1, col2 = st.columns([0.4, 0.6]) # Ajuste nas proporções das colunas

with col1:
    st.subheader("🛠️ Parâmetros de Entrada")

    # --- Seleção do Método de Entrada ---
    input_method = st.radio(
        "Método de Entrada da Função:",
        ("Funções Pré-definidas", "Função Personalizada"), # Invertido para padrão mais comum
        horizontal=True,
        key="input_method_radio"
    )

    func_str = ""

    # --- Entrada Condicional da Função ---
    if input_method == "Função Personalizada":
        st.markdown("""
        **Instruções:** Digite sua função $z = f(x, y)$.
        * **Variáveis:** `x`, `y`, `pi`, `size`.
        * **Funções:** `sin`, `cos`, `tan`, `sqrt`, `exp`, `log`, `log10`, `abs`, `cosh`, `sinh`, `tanh`, `power`.
        * **Exemplo:** `power(x, 2) + power(y, 2)` para $x^2 + y^2$.
        * ⚠️ **Atenção:** Funções complexas ou mal formadas podem causar erros ou lentidão. Use por sua conta e risco.
        """)
        default_custom_func = st.session_state.get("last_custom_func", "sin(sqrt(x**2 + y**2))")
        func_str_custom = st.text_input(
            "Função $z = f(x, y)$:",
            value=default_custom_func,
            key="custom_func_input",
            placeholder="Ex: 0.5 * (x**2 + y**2)"
            )
        if func_str_custom:
            func_str = func_str_custom
            st.session_state["last_custom_func"] = func_str
        else:
             st.warning("Por favor, insira uma função personalizada.")

    else: # "Funções Pré-definidas"
        default_predefined_key = st.session_state.get("last_predefined_func_name", "Ondas Circulares")
        # Garante que a chave padrão ainda exista no dicionário
        if default_predefined_key not in predefined_functions:
             default_predefined_key = list(predefined_functions.keys())[0]

        # Calcula o índice da chave padrão
        options_list = list(predefined_functions.keys())
        try:
            default_predefined_index = options_list.index(default_predefined_key)
        except ValueError:
             default_predefined_index = 0 # Se não encontrar, usa o primeiro

        selected_func_name = st.selectbox(
            "Escolha uma função pré-definida:",
            options=options_list,
            index=default_predefined_index,
            key="predefined_func_select"
        )
        func_str = predefined_functions[selected_func_name]
        st.session_state["last_predefined_func_name"] = selected_func_name
        st.code(f"f(x, y) = {func_str}", language="python")

    # --- Sliders de Parâmetros ---
    st.divider()
    res = st.slider("Resolução (Detalhes):", 4, 500, 150, step=10, help="Qualidade da malha. Valores altos exigem mais processamento.", key="res_slider")
    ampl = st.slider("Amplitude (Altura Z):", 0.1, 20.0, 1.0, step=0.1, help="Multiplicador da altura Z final.", key="ampl_slider")
    size = st.slider("Tamanho da Base (X/Y):", 1.0, 50.0, 10.0, step=0.5, help="Define o intervalo [-tamanho, tamanho] para X e Y.", key="size_slider")

    st.divider()
    gerar_btn = st.button("🚀 Gerar Modelo 3D e Visualizar", type="primary", use_container_width=True)
    st.divider()

    # --- Links e Informações Adicionais ---
    st.markdown("""
    **Visualizador Externo:**
    Para inspecionar o arquivo `.stl` gerado em detalhes antes de imprimir:
    """)
    st.link_button("Abrir 3DViewer.net", url='https://3dviewer.net/', use_container_width=True)

    st.caption("Desenvolvido com Streamlit e NumPy-STL.")


with col2:
    st.subheader("📊 Visualização e Download 💾")
    plot_placeholder = st.container() # Usar container para melhor controle
    download_placeholder = st.container()

    if gerar_btn:
        if not func_str:
            st.error("⚠️ Por favor, digite ou selecione uma função matemática válida na coluna à esquerda.")
        else:
            # Mostrar spinner enquanto calcula e salva
            with st.spinner("⚙️ Calculando malha e gerando visualização..."):
                # 1. Calcular a malha (pode usar cache)
                mesh_data, error_calc = calcular_malha(res, ampl, size, func_str)

                if error_calc:
                    st.error(f"❌ Erro no Cálculo: {error_calc}")
                    plot_placeholder.empty()
                    download_placeholder.empty()
                elif mesh_data:
                    x, y, z, all_vertices, faces, x_vals_ret, y_vals_ret = mesh_data

                    # --- Visualização 3D ---
                    try:
                        fig = plt.figure(figsize=(10, 8)) # Ajuste o tamanho
                        ax = fig.add_subplot(111, projection='3d')
                        # Usar stride para plotar menos pontos na visualização se a resolução for muito alta
                        stride_val = max(1, res // 100) # Ex: plota 1 a cada N pontos se res > 100
                        surf = ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none', antialiased=True, rstride=stride_val, cstride=stride_val)
                        # Adiciona base visual (opcional, pode poluir)
                        # xx, yy = np.meshgrid(x_vals_ret, y_vals_ret)
                        # ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.15, color='gray')

                        ax.set_title(f"Visualização Aprox.: {func_str[:50]}{'...' if len(func_str)>50 else ''}", fontsize=11)
                        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
                        max_z_val = np.max(z)
                        ax.set_zlim(0, max_z_val * 1.1 if max_z_val > 0 else 1)
                        ax.view_init(elev=25, azim=-110) # Ângulo de visão
                        # Adicionar barra de cores se não for muito poluído
                        # fig.colorbar(surf, shrink=0.5, aspect=10, label="Altura Z")
                        plt.tight_layout(pad=0.5)
                        plot_placeholder.pyplot(fig)
                        plt.close(fig) # Libera memória do plot
                        st.success("✅ Visualização gerada!")

                    except Exception as e_plot:
                         st.warning(f"⚠️ Erro ao gerar visualização: {e_plot}. O arquivo STL ainda pode estar correto.")
                         print(f"Erro detalhado ao plotar:")
                         print(traceback.format_exc())
                         plot_placeholder.empty() # Limpa se o plot falhar


                    # --- Geração e Download do STL ---
                    with st.spinner("💾 Preparando arquivo STL para download..."):
                        # 2. Salvar a malha em STL (fora do cache)
                        stl_path, error_save = salvar_stl(all_vertices, faces, file_prefix=f"func_{input_method[:4]}")

                        if error_save:
                             st.error(f"❌ Erro ao Salvar STL: {error_save}")
                             download_placeholder.empty()
                        elif stl_path and os.path.exists(stl_path):
                            try:
                                with open(stl_path, 'rb') as f:
                                    stl_bytes = f.read()

                                # Gera um nome de arquivo mais seguro
                                safe_func_name = "".join(c if c.isalnum() else "_" for c in func_str[:25]).rstrip('_')
                                file_name_stl = f"modelo_{safe_func_name}_r{res}_s{size}.stl"

                                download_placeholder.download_button(
                                    label=f"📥 Baixar: {file_name_stl}",
                                    data=stl_bytes,
                                    file_name=file_name_stl,
                                    mime="model/stl",
                                    use_container_width=True
                                )
                                st.success("✅ Arquivo STL pronto para download!")

                            except FileNotFoundError:
                                st.error(f"Erro Crítico: O arquivo temporário {stl_path} não foi encontrado após ser criado.")
                                download_placeholder.empty()
                            except Exception as e_down:
                                st.error(f"Erro ao preparar o download: {e_down}")
                                download_placeholder.empty()
                            finally:
                                # Tenta remover o arquivo temporário APÓS o botão ser preparado/clicado
                                # Idealmente, a remoção ocorreria após o download, mas Streamlit não facilita isso.
                                # Remover aqui significa que o arquivo pode sumir se o usuário demorar para clicar.
                                # Uma alternativa é agendar a limpeza ou usar um diretório temporário gerenciado.
                                if stl_path and os.path.exists(stl_path):
                                    try:
                                        # Não remover imediatamente para dar tempo ao usuário baixar
                                        # A limpeza de arquivos temporários do sistema geralmente cuida disso.
                                        # os.remove(stl_path)
                                        pass # Deixar o sistema operacional limpar /tmp
                                    except OSError as e_rem:
                                        st.warning(f"Não foi possível remover o arquivo temporário {stl_path}: {e_rem}")
                        else:
                             st.error("❌ Falha desconhecida ao gerar o caminho do arquivo STL.")
                             download_placeholder.empty()
            # Fim do bloco try/except principal
    else:
        # Mensagem inicial ou quando não está processando
        plot_placeholder.info("Configure os parâmetros à esquerda e clique em 'Gerar Modelo 3D' para visualizar e baixar o arquivo STL.")
        download_placeholder.empty() # Garante que o botão de download não fique visível


