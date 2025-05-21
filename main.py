import streamlit as st
import numpy as np
import tempfile
from stl import mesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import traceback
import matplotlib.tri as mtri

# --- Configurações Iniciais ---
st.set_page_config(
    layout="wide",
    page_title="Gerador 3D por Função",
    page_icon="🧊"
)


# --- Funções de Geração da Malha (Cacheada) ---
@st.cache_data(max_entries=50)
def calcular_malha(res, ampl, size, func_str, modo="função"):
    """
    Calcula os vértices e faces da malha 3D com base nos parâmetros.
    Esta função é cacheada para performance.
    """
    if res < 4:
        return None, "Erro: A resolução deve ser pelo menos 4."

    try:
        if modo == "hiperboloide":
            u_count = min(res, 100)
            v_count = min(max(res // 5, 10), 40)
            v_range = size / 3.0
            a_param = ampl

            u_vals = np.linspace(0, 2.0 * np.pi, u_count, endpoint=True)
            v_vals = np.linspace(-v_range, v_range, v_count, endpoint=True)
            u_grid, v_grid = np.meshgrid(u_vals, v_vals)

            x_plot = a_param * np.cosh(v_grid) * np.cos(u_grid)
            y_plot = a_param * np.cosh(v_grid) * np.sin(u_grid)
            z_plot = a_param * np.sinh(v_grid)

            u_flat = u_grid.flatten()
            v_flat = v_grid.flatten()
            x_flat_stl = a_param * np.cosh(v_flat) * np.cos(u_flat)
            y_flat_stl = a_param * np.cosh(v_flat) * np.sin(u_flat)
            z_flat_stl = a_param * np.sinh(v_flat)
            all_vertices = np.column_stack((x_flat_stl, y_flat_stl, z_flat_stl))

            tri = mtri.Triangulation(u_flat, v_flat)
            faces = tri.triangles

            x_vals_dummy = np.linspace(-size, size, res)
            y_vals_dummy = np.linspace(-size, size, res)
            return (x_plot, y_plot, z_plot, all_vertices, faces, x_vals_dummy, y_vals_dummy), None

        else:  # Modo função cartesiana
            x_vals = np.linspace(-size, size, res)
            y_vals = np.linspace(-size, size, res)
            x_grid, y_grid = np.meshgrid(x_vals, y_vals)

            safe_dict = {
                "x": x_grid, "y": y_grid, "np": np,
                "sin": np.sin, "cos": np.cos, "tan": np.tan,
                "sqrt": np.sqrt, "exp": np.exp, "log": np.log, "log10": np.log10,
                "abs": np.abs, "pi": np.pi, "size": size,
                "cosh": np.cosh, "sinh": np.sinh, "tanh": np.tanh,
                "power": np.power,
            }

            z_eval_raw = eval(func_str, {"__builtins__": None}, safe_dict)

            if not isinstance(z_eval_raw, np.ndarray) or not np.issubdtype(z_eval_raw.dtype, np.number):
                raise ValueError("A função não retornou um resultado numérico (array NumPy) válido.")

            z_eval_ampl = ampl * z_eval_raw
            invalid_mask = np.isnan(z_eval_ampl) | np.isinf(z_eval_ampl)
            num_invalid = np.sum(invalid_mask)

            z_processed = np.copy(z_eval_ampl)
            if num_invalid > 0:
                st.warning(
                    f"Atenção: A função '{func_str}' produziu {num_invalid} valores indefinidos (NaN) ou infinitos. "
                    f"Estes pontos foram mapeados para Z=0 (relativo à base da função) para a geração da malha."
                )
                z_processed[invalid_mask] = 0.0

            min_z_val = np.min(z_processed)
            z_plot = z_processed - min_z_val

            # MODIFICAÇÃO PARA OTIMIZAÇÃO (Menos "Duro"):
            # Forçar valores muito próximos de zero a serem zero de forma mais sutil.
            # Isso ajuda a limpar as bordas, mas com um epsilon menor para não achatar demais.
            effective_range_z = np.max(z_plot)  # min_z_plot é 0 após a normalização
            if effective_range_z > 1e-9:
                # Epsilon bem pequeno, relativo à altura efetiva da superfície.
                # 1e-6 deve ser sutil o suficiente para não achatar características visíveis.
                epsilon_z = 1e-6 * effective_range_z

                mask_to_flatten = (z_plot > 0) & (z_plot < epsilon_z)  # Apenas positivos pequenos
                if np.any(mask_to_flatten):
                    # st.caption(f"Debug: {np.sum(mask_to_flatten)} pontos superficialmente achatados para Z=0 (epsilon={epsilon_z:.2e})")
                    z_plot[mask_to_flatten] = 0.0

            x_flat, y_flat, z_flat_top = x_grid.flatten(), y_grid.flatten(), z_plot.flatten()
            num_vertices_per_layer = res * res
            vertices_top = np.column_stack((x_flat, y_flat, z_flat_top))
            base_z_values = np.zeros_like(z_flat_top)
            vertices_base = np.column_stack((x_flat, y_flat, base_z_values))
            all_vertices = np.vstack((vertices_top, vertices_base))

            faces = []
            # 1. Faces Superiores (normais para cima: +Z)
            for i in range(res - 1):
                for j in range(res - 1):
                    v_ij = i * res + j
                    v_ij1 = v_ij + 1
                    v_i1j = v_ij + res
                    v_i1j1 = v_i1j + 1
                    faces.append([v_ij, v_i1j, v_ij1])
                    faces.append([v_ij1, v_i1j, v_i1j1])

            # 2. Faces Inferiores (normais para baixo: -Z)
            offset = num_vertices_per_layer
            for i in range(res - 1):
                for j in range(res - 1):
                    v_ij_b = i * res + j + offset
                    v_ij1_b = v_ij_b + 1
                    v_i1j_b = v_ij_b + res
                    v_i1j1_b = v_i1j_b + 1
                    faces.append([v_ij_b, v_ij1_b, v_i1j_b])  # Ordem invertida
                    faces.append([v_ij1_b, v_i1j1_b, v_i1j_b])  # Ordem invertida

            # 3. Faces Laterais (normais para fora)
            for i in range(res - 1):  # Iterar ao longo das linhas (y varia, x constante nas bordas E/D)
                # Borda Esquerda (x = x_min), normal para -X
                idx_t_curr = i * res
                idx_t_next = (i + 1) * res
                idx_b_curr = idx_t_curr + offset
                idx_b_next = idx_t_next + offset
                faces.append([idx_t_curr, idx_b_next, idx_b_curr])  # P1_top, P2_base, P1_base
                faces.append([idx_t_curr, idx_t_next, idx_b_next])  # P1_top, P2_top, P2_base

                # Borda Direita (x = x_max), normal para +X
                idx_t_curr = i * res + (res - 1)
                idx_t_next = (i + 1) * res + (res - 1)
                idx_b_curr = idx_t_curr + offset
                idx_b_next = idx_t_next + offset
                faces.append([idx_t_curr, idx_b_curr, idx_b_next])  # P1_top, P1_base, P2_base
                faces.append([idx_t_curr, idx_b_next,
                              idx_t_next])  # P1_top, P2_base, P2_top (ordem para manter aresta idx_t_curr,idx_b_next)

            for j in range(res - 1):  # Iterar ao longo das colunas (x varia, y constante nas bordas Inf/Sup)
                # Borda Inferior (y = y_min), normal para -Y
                idx_t_curr = j
                idx_t_next = j + 1
                idx_b_curr = idx_t_curr + offset
                idx_b_next = idx_t_next + offset
                faces.append([idx_t_curr, idx_b_next, idx_b_curr])
                faces.append([idx_t_curr, idx_t_next, idx_b_next])

                # Borda Superior (y = y_max), normal para +Y
                idx_t_curr = (res - 1) * res + j
                idx_t_next = (res - 1) * res + j + 1
                idx_b_curr = idx_t_curr + offset
                idx_b_next = idx_t_next + offset
                faces.append([idx_t_curr, idx_b_curr, idx_b_next])
                faces.append([idx_t_curr, idx_b_next, idx_t_next])

            return (x_grid, y_grid, z_plot, all_vertices, faces, x_vals, y_vals), None

    except Exception as e:
        print(f"Erro detalhado em calcular_malha para func='{func_str}', modo='{modo}':")
        print(traceback.format_exc())
        user_error_message = f"Erro ao processar '{func_str if func_str else modo}': {str(e)[:200]}."
        return None, user_error_message


# --- Função para Salvar STL (Não Cacheada) ---
def salvar_stl(all_vertices, faces, file_prefix="modelo_stl"):
    try:
        num_faces = len(faces)
        if num_faces == 0: return None, "Erro: Nenhuma face gerada para criar o STL."
        if all_vertices is None or all_vertices.shape[0] == 0: return None, "Erro: Nenhum vértice gerado."

        nan_mask_verts = np.isnan(all_vertices)
        inf_mask_verts = np.isinf(all_vertices)
        if np.any(nan_mask_verts) or np.any(inf_mask_verts):
            st.warning("Aviso STL: Vértices continham NaN/Inf e foram substituídos. O STL pode ter artefatos.")
            all_vertices = np.nan_to_num(all_vertices, nan=0.0, posinf=1e6, neginf=-1e6)

        solid_mesh_data = np.zeros(num_faces, dtype=mesh.Mesh.dtype)
        valid_face_count = 0
        for i, face_indices in enumerate(faces):
            if any(idx < 0 or idx >= len(all_vertices) for idx in face_indices):
                continue
            face_verts = all_vertices[face_indices]
            if np.any(np.isnan(face_verts)) or np.any(np.isinf(face_verts)):
                continue
            solid_mesh_data['vectors'][valid_face_count] = face_verts
            valid_face_count += 1

        if valid_face_count == 0:
            return None, "Erro: Nenhuma face válida restante após a limpeza para criar o STL."
        solid_mesh = mesh.Mesh(solid_mesh_data[:valid_face_count])

        with tempfile.NamedTemporaryFile(delete=False, suffix='.stl', prefix=file_prefix + "_") as temp_file:
            solid_mesh.save(temp_file.name)
            temp_file_path = temp_file.name
        return temp_file_path, None
    except Exception as e:
        print(f"Erro detalhado ao salvar STL:");
        print(traceback.format_exc())
        return None, f"Erro ao criar ou salvar o arquivo STL: {e}"


# === Interface Streamlit ===
st.title("🧊 Gerador de Modelo 3D (STL) por Função Matemática")
st.markdown("""
Crie modelos 3D sólidos e prontos para impressão FDM (.stl) inserindo uma função matemática $z = f(x, y)$ ou escolhendo uma da lista.
**Nota:** 
- A visualização é uma aproximação. Use o visualizador externo para o modelo exato.
- Se a função $z=f(x,y)$ resultar em valores indefinidos (ex: `sqrt(-1)`), esses pontos serão mapeados para $Z=0$ (relativo à base da função) na malha.
- Funções com descontinuidades ou derivadas muito íngremes nas bordas de seus domínios podem gerar artefatos na malha. Aumentar a **Resolução** ou usar um software de reparo de STL externo pode ser necessário para otimizar a impressão.
""")
st.markdown("---")

predefined_functions = {
    "Ondas Circulares": "sin(sqrt(x**2 + y**2))",
    "Sela": "0.2 * (x**2 - y**2)",
    "Vulcão Simples": "exp(-(x**2 + y**2) / (0.5*size)) * size",
    "Grades Cruzadas": "cos(x) * sin(y)",
    "Picos Senoidais": "sin(pi*x/size) * sin(pi*y/size) * size/2",
    "Paraboloide": "0.1 * (x**2 + y**2)",
    "Função Teste (com sqrt)": "sqrt(x**2 + y**2 + 1)",
    "Função com Borda (sqrt)": "sqrt(x**2 - y**2 - 16)",  # Exemplo do usuário
    "Ripples": "sin(10 * sqrt(x**2 + y**2)) / 10",
    "Função 'Peaks' MATLAB": "3*(1-x)**2*exp(-(x**2) - (y+1)**2) - 10*(x/5 - x**3 - y**5)*exp(-x**2-y**2) - 1/3*exp(-(x+1)**2 - y**2)",
    "Terreno Fractal Simples": "0.3*sin(x/2)*cos(y/3) + 0.2*sin(x*2)*cos(y*1.5) + 0.1*sin(x*5)*cos(y*4)",
    "Hiperboloide (Implícita, problemática)": "1/sqrt(power(y,2)-power(x,2) + 0.01)"
}

col1, col2 = st.columns([0.4, 0.6])

with col1:
    st.subheader("🛠️ Parâmetros de Entrada")
    input_method = st.radio(
        "Método de Entrada da Função:",
        ("Funções Pré-definidas", "Função Personalizada", "Hiperboloide Paramétrico"),
        horizontal=True, key="input_method_radio"
    )

    func_str = ""
    modo_calculo = "função"

    if input_method == "Hiperboloide Paramétrico":
        st.markdown("""
        **Hiperboloide Paramétrico de Uma Folha** 
        - x = A * cosh(v) * cos(u)
        - y = A * cosh(v) * sin(u)
        - z = A * sinh(v)
        'A' é 'Amplitude', 'v' varia com 'Tamanho da Base'.
        """)
        func_str = "HIPERBOLOIDE_PARAM"
        modo_calculo = "hiperboloide"
    elif input_method == "Função Personalizada":
        st.markdown("""
        **Instruções:** $z = f(x, y)$.
        * **Variáveis:** `x`, `y`, `pi`, `size`.
        * **Funções:** `sin`, `cos`, `tan`, `sqrt`, `exp`, `log`, `log10`, `abs`, `cosh`, `sinh`, `tanh`, `power`.
        * **Exemplo:** `power(x, 2) + power(y, 2)`.
        """)
        default_custom_func = st.session_state.get("last_custom_func", "sqrt(x**2 - y**2 - 16)")
        func_str_custom = st.text_input(
            "Função $z = f(x, y)$:", value=default_custom_func,
            key="custom_func_input", placeholder="Ex: 0.5 * (x**2 + y**2)"
        )
        if func_str_custom:
            func_str = func_str_custom
            st.session_state["last_custom_func"] = func_str
        else:
            st.warning("Por favor, insira uma função personalizada.")
    else:  # "Funções Pré-definidas"
        default_predefined_key = st.session_state.get("last_predefined_func_name", "Função com Borda (sqrt)")
        if default_predefined_key not in predefined_functions:
            default_predefined_key = list(predefined_functions.keys())[0]
        options_list = list(predefined_functions.keys())
        try:
            default_predefined_index = options_list.index(default_predefined_key)
        except ValueError:
            default_predefined_index = 0
        selected_func_name = st.selectbox(
            "Escolha uma função pré-definida:", options=options_list,
            index=default_predefined_index, key="predefined_func_select"
        )
        func_str = predefined_functions[selected_func_name]
        st.session_state["last_predefined_func_name"] = selected_func_name
        st.code(f"f(x, y) = {func_str}", language="python")

    st.divider()
    res = st.slider("Resolução (Detalhes):", 4, 350, 150, step=2, key="res_slider",
                    help="Maior resolução pode melhorar bordas de funções complexas, mas aumenta o processamento.")  # Aumentei um pouco o max
    ampl = st.slider("Amplitude (Altura Z / Parâmetro A):", 0.1, 20.0, 5.0, step=0.1, key="ampl_slider")
    size = st.slider("Tamanho da Base (X/Y / Escala v):", 1.0, 30.0, 10.0, step=0.5, key="size_slider")
    st.divider()
    gerar_btn = st.button("🚀 Gerar Modelo 3D e Visualizar", type="primary", use_container_width=True)
    st.divider()
    st.markdown("**Visualizador Externo:**")
    st.link_button("Abrir 3DViewer.net", url='https://3dviewer.net/', use_container_width=True)
    st.caption("Desenvolvido com Streamlit e NumPy-STL.")

with col2:
    st.subheader("📊 Visualização e Download 💾")
    plot_placeholder = st.container()
    download_placeholder = st.container()

    if gerar_btn:
        if (
                not func_str or func_str == "HIPERBOLOIDE_PARAM" and modo_calculo != "hiperboloide") and modo_calculo == "função":
            st.error("⚠️ Por favor, digite ou selecione uma função matemática válida na coluna à esquerda.")
        else:
            with st.spinner("⚙️ Calculando malha e gerando visualização..."):
                mesh_data, error_calc = calcular_malha(res, ampl, size, func_str, modo=modo_calculo)

                if error_calc:
                    st.error(f"❌ Erro no Cálculo: {error_calc}")
                    plot_placeholder.empty();
                    download_placeholder.empty()
                elif mesh_data:
                    x_plot, y_plot, z_plot, all_vertices, faces, _, _ = mesh_data

                    try:
                        fig = plt.figure(figsize=(10, 8))
                        ax = fig.add_subplot(111, projection='3d')
                        title_str = func_str if modo_calculo == "função" else "Hiperboloide Paramétrico"

                        r_stride = max(1, x_plot.shape[0] // 75 if x_plot.ndim == 2 and x_plot.shape[0] > 0 else 1)
                        c_stride = max(1, x_plot.shape[1] // 75 if x_plot.ndim == 2 and x_plot.shape[1] > 0 else 1)

                        surf = ax.plot_surface(x_plot, y_plot, z_plot, cmap='viridis', edgecolor='none',
                                               antialiased=True, rstride=r_stride, cstride=c_stride)

                        ax.set_title(f"Visualização: {title_str[:50]}{'...' if len(title_str) > 50 else ''}",
                                     fontsize=11)
                        ax.set_xlabel("X");
                        ax.set_ylabel("Y");
                        ax.set_zlabel("Z")

                        if modo_calculo == "hiperboloide":
                            coords = np.concatenate([c.flatten() for c in [x_plot, y_plot, z_plot] if c is not None])
                            finite_coords = coords[np.isfinite(coords)]
                            abs_max = np.max(np.abs(finite_coords)) if finite_coords.size > 0 else 1.0
                            ax.set_xlim(-abs_max, abs_max);
                            ax.set_ylim(-abs_max, abs_max);
                            ax.set_zlim(-abs_max, abs_max)
                            ax.view_init(elev=20, azim=30)
                        else:
                            finite_z_plot = z_plot[np.isfinite(z_plot)]
                            max_z_val_plot = np.max(finite_z_plot) if finite_z_plot.size > 0 else 1.0
                            min_z_val_plot = np.min(finite_z_plot) if finite_z_plot.size > 0 else 0.0
                            plot_range_z = max_z_val_plot - min_z_val_plot
                            if plot_range_z < 1e-3: plot_range_z = 1.0
                            ax.set_zlim(min_z_val_plot - 0.05 * plot_range_z,
                                        max_z_val_plot + 0.1 * plot_range_z)  # Aumentei um pouco a margem superior
                            ax.view_init(elev=25, azim=-110)

                        plt.tight_layout(pad=0.5)
                        plot_placeholder.pyplot(fig)
                        plt.close(fig)
                        st.success("✅ Visualização gerada!")
                    except Exception as e_plot:
                        st.warning(f"⚠️ Erro ao gerar visualização: {e_plot}. O arquivo STL ainda pode ser gerado.")
                        print(f"Erro detalhado ao plotar:");
                        print(traceback.format_exc())
                        plot_placeholder.empty()

                    with st.spinner("💾 Preparando arquivo STL para download..."):
                        file_prefix_stl = "hiperboloide" if modo_calculo == "hiperboloide" else f"func_{input_method[:4]}"
                        stl_path, error_save = salvar_stl(all_vertices, faces, file_prefix=file_prefix_stl)

                        if error_save:
                            st.error(f"❌ Erro ao Salvar STL: {error_save}")
                            download_placeholder.empty()
                        elif stl_path and os.path.exists(stl_path):
                            try:
                                with open(stl_path, 'rb') as f:
                                    stl_bytes = f.read()
                                if modo_calculo == "hiperboloide":
                                    file_name_stl = f"hiperboloide_r{res}_a{ampl}_s{size}.stl"
                                else:
                                    safe_func_name = "".join(c if c.isalnum() else "_" for c in func_str[:20]).rstrip(
                                        '_')
                                    file_name_stl = f"modelo_{safe_func_name}_r{res}_s{size}.stl"
                                download_placeholder.download_button(
                                    label=f"📥 Baixar: {file_name_stl}", data=stl_bytes,
                                    file_name=file_name_stl, mime="model/stl", use_container_width=True
                                )
                                st.success("✅ Arquivo STL pronto para download!")
                            except Exception as e_down:
                                st.error(f"Erro ao preparar o download: {e_down}");
                                download_placeholder.empty()
                        else:
                            st.error("❌ Falha desconhecida ao gerar o caminho do STL.");
                            download_placeholder.empty()
    else:
        plot_placeholder.info("Configure os parâmetros e clique em 'Gerar Modelo 3D' para visualizar e baixar.")
        download_placeholder.empty()