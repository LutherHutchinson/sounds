# pyrefly: ignore [missing-import]
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import os
from scipy.io import wavfile
# pyrefly: ignore [missing-import]
import plotly.graph_objects as go
# pyrefly: ignore [missing-import]
import plotly.express as px

# Local imports
import thinkdsp
from thinkdsp import (
    Chirp, UncorrelatedUniformNoise, BrownianNoise, PinkNoise,
    UncorrelatedGaussianNoise, SquareSignal, CosSignal, SinSignal,
    Noise, Sinusoid, normalize, unbias, Wave,
    UncorrelatedPoissonNoise, SawtoothSignal, SawtoothChirp
)

# Page configuration
st.set_page_config(
    page_title="ПРИЛОЖЕНИЕ ПО ЦОС",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS for light theme and glassmorphic cards
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Outfit:wght@400;600;800&display=swap');
        
        /* Font styling */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            background-color: #fff0f3;
            color: #3a0f28;
        }
        
        /* Main title styling */
        .main-header {
            font-family: 'Outfit', sans-serif;
            background: linear-gradient(135deg, #ff3377 0%, #ff85a2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.8rem;
            font-weight: 800;
            margin-bottom: 0.1rem;
            text-align: center;
        }
        .subheader {
            color: #e63970;
            font-size: 1.1rem;
            margin-bottom: 2rem;
            text-align: center;
            font-weight: 400;
        }
        
        /* Metric cards styling */
        .metric-card {
            background-color: #ffffff;
            border: 1px solid #ffd1dc;
            border-radius: 12px;
            padding: 1.2rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 10px rgba(255, 133, 162, 0.1);
            transition: transform 0.2s, border-color 0.2s, box-shadow 0.2s;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            border-color: #ff3377;
            box-shadow: 0 6px 20px rgba(255, 51, 119, 0.18);
        }
        .metric-title {
            color: #c24177;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.3rem;
            font-weight: 600;
        }
        .metric-value {
            color: #3a0f28;
            font-size: 1.6rem;
            font-weight: 700;
        }
        .metric-desc {
            color: #ff3377;
            font-size: 0.8rem;
            margin-top: 0.2rem;
        }
        
        /* Sidebar layout styling */
        section[data-testid="stSidebar"] {
            background-color: #ffe6ec;
            border-right: 1px solid #ffb3c1;
        }
        
        /* Style Streamlit buttons */
        div.stButton > button:first-child {
            background: linear-gradient(135deg, #ff3377 0%, #ff6699 100%);
            color: #ffffff;
            font-weight: 700;
            border: none;
            padding: 0.6rem 2rem;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(255, 51, 119, 0.25);
            transition: all 0.2s;
        }
        div.stButton > button:first-child:hover {
            transform: scale(1.02);
            box-shadow: 0 6px 16px rgba(255, 51, 119, 0.35);
            color: #ffffff;
        }
        
        /* Card sections */
        .glass-card, div[data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255, 255, 255, 0.85);
            border: 1px solid rgba(255, 179, 193, 0.5) !important;
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(255, 133, 162, 0.08);
        }
    </style>
""", unsafe_allow_html=True)

# Set matplotlib properties to match Streamlit's light theme
def setup_plot_style():
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = '#fff0f3'
    plt.rcParams['axes.facecolor'] = '#ffffff'
    plt.rcParams['axes.edgecolor'] = '#ffd1dc'
    plt.rcParams['grid.color'] = '#ffe3e8'
    plt.rcParams['grid.alpha'] = 0.5
    plt.rcParams['text.color'] = '#3a0f28'
    plt.rcParams['axes.labelcolor'] = '#8b3a62'
    plt.rcParams['xtick.color'] = '#8b3a62'
    plt.rcParams['ytick.color'] = '#8b3a62'
    plt.rcParams['font.family'] = 'sans-serif'

setup_plot_style()

def make_plotly_line(x, y, title, x_title, y_title, line_color='#ff4d88', is_scatter=False):
    fig = go.Figure()
    if is_scatter:
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(color=line_color, size=4), hoverinfo='x+y'))
    else:
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color=line_color, width=1.5), hoverinfo='x+y'))
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=14, family='Outfit, sans-serif', color='#3a0f28', weight='bold'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title=dict(text=x_title, font=dict(size=11, color='#8b3a62')),
            tickfont=dict(size=9, color='#8b3a62'),
            gridcolor='#ffe3e8',
            linecolor='#ffb3c1'
        ),
        yaxis=dict(
            title=dict(text=y_title, font=dict(size=11, color='#8b3a62')),
            tickfont=dict(size=9, color='#8b3a62'),
            gridcolor='#ffe3e8',
            linecolor='#ffb3c1'
        ),
        margin=dict(l=45, r=15, t=45, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#ffffff',
        hovermode='closest',
        showlegend=False
    )
    return fig

def make_plotly_spectrogram(sp, title, high_freq=None):
    fs = sp.frequencies()
    i = None if high_freq is None else thinkdsp.find_index(high_freq, fs)
    fs_cut = fs[:i]
    ts = sp.times()
    
    # Get 2D amplitude grid
    size = len(fs_cut), len(ts)
    z_data = np.zeros(size, dtype=float)
    for col_idx, t in enumerate(ts):
        spectrum = sp.spec_map[t]
        z_data[:, col_idx] = spectrum.amps[:i]
        
    fig = go.Figure(data=go.Heatmap(
        x=ts,
        y=fs_cut,
        z=z_data,
        colorscale='Plasma',
        hoverongaps=False
    ))
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=14, family='Outfit, sans-serif', color='#3a0f28', weight='bold'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(title=dict(text='Время (с)', font=dict(color='#8b3a62')), tickfont=dict(color='#8b3a62')),
        yaxis=dict(title=dict(text='Частота (Гц)', font=dict(color='#8b3a62')), tickfont=dict(color='#8b3a62')),
        margin=dict(l=45, r=15, t=45, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#ffffff'
    )
    return fig

def make_plotly_overlaid_lines(x_a, y_a, label_a, x_b, y_b, label_b, title, x_title, y_title, color_a='#ff4d88', color_b='#9333ea', is_dashed_b=False):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_a, y=y_a, mode='lines', name=label_a, line=dict(color=color_a, width=1.5)))
    line_dict_b = dict(color=color_b, width=1.5)
    if is_dashed_b:
        line_dict_b['dash'] = 'dash'
    fig.add_trace(go.Scatter(x=x_b, y=y_b, mode='lines', name=label_b, line=line_dict_b))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, family='Outfit, sans-serif', color='#3a0f28', weight='bold'), x=0.5, xanchor='center'),
        xaxis=dict(title=dict(text=x_title, font=dict(color='#8b3a62')), tickfont=dict(color='#8b3a62'), gridcolor='#ffe3e8', linecolor='#ffb3c1'),
        yaxis=dict(title=dict(text=y_title, font=dict(color='#8b3a62')), tickfont=dict(color='#8b3a62'), gridcolor='#ffe3e8', linecolor='#ffb3c1'),
        margin=dict(l=45, r=15, t=45, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#ffffff',
        hovermode='closest',
        showlegend=True,
        legend=dict(x=0.85, y=0.95, bgcolor='rgba(255,255,255,0.7)', bordercolor='#ffb3c1', borderwidth=1)
    )
    return fig


# --- Helper DSP functions ---

def get_signal_object(signal_type, start_freq, end_freq):
    if signal_type == 'Chirp':
        return Chirp(start=start_freq, end=end_freq)
    elif signal_type == 'SawtoothChirp':
        return SawtoothChirp(start=start_freq, end=end_freq)
    elif signal_type == 'UncorrelatedUniformNoise':
        return UncorrelatedUniformNoise()
    elif signal_type == 'BrownianNoise':
        return BrownianNoise()
    elif signal_type == 'PinkNoise':
        return PinkNoise()
    elif signal_type == 'UncorrelatedGaussianNoise':
        return UncorrelatedGaussianNoise()
    elif signal_type == 'UncorrelatedPoissonNoise':
        return UncorrelatedPoissonNoise()
    elif signal_type == 'SawtoothSignal':
        return SawtoothSignal(freq=start_freq)
    elif signal_type == 'SquareSignal':
        return SquareSignal(freq=start_freq)
    elif signal_type == 'CosSignal':
        return CosSignal(freq=start_freq)
    elif signal_type == 'SinSignal':
        return SinSignal(freq=start_freq)
    else:
        raise ValueError(f"Неизвестный тип сигнала: {signal_type}")

def load_wav_from_bytes(uploaded_file):
    bytes_data = uploaded_file.read()
    framerate, ys = wavfile.read(io.BytesIO(bytes_data))
    
    # Stereo to mono conversion
    if ys.ndim == 2:
        ys = ys[:, 0]
        
    # Scale PCM integer files to [-1.0, 1.0]
    if ys.dtype == np.int16:
        ys = ys.astype(np.float64) / 32768.0
    elif ys.dtype == np.int32:
        ys = ys.astype(np.float64) / 2147483648.0
    elif ys.dtype == np.uint8:
        ys = (ys.astype(np.float64) - 128.0) / 128.0
    else:
        ys = ys.astype(np.float64)
        
    # De-bias and normalize
    ys = normalize(unbias(ys))
    
    # Duration check and warning/truncation for performance safety
    duration = len(ys) / framerate
    if duration > 5.0:
        st.sidebar.warning(f"Файл обрезан до первых 5 секунд (исходно: {duration:.2f} сек) для быстрой обработки.")
        ys = ys[:int(5.0 * framerate)]
        
    return Wave(ys, framerate=framerate)

def make_audio_bytes(wave):
    # Safe normalization to prevent clipping in browser players
    ys_norm = wave.ys / max(np.max(np.abs(wave.ys)), 1e-5)
    ys_int = (ys_norm * 32767).astype(np.int16)
    buffer = io.BytesIO()
    wavfile.write(buffer, wave.framerate, ys_int)
    return buffer.getvalue()

def align_waves(wave_a, wave_b):
    # Align waves to have the same length and sample rate for comparison
    # We crop/segment to the minimum duration, and use the higher framerate for alignment quality
    min_dur = min(wave_a.duration, wave_b.duration)
    target_framerate = max(wave_a.framerate, wave_b.framerate)
    
    w1_seg = wave_a.segment(start=0, duration=min_dur)
    w2_seg = wave_b.segment(start=0, duration=min_dur)
    
    n_samples = int(round(min_dur * target_framerate))
    ts_common = np.arange(n_samples) / target_framerate
    
    ys1_interp = np.interp(ts_common, w1_seg.ts - w1_seg.start, w1_seg.ys)
    ys2_interp = np.interp(ts_common, w2_seg.ts - w2_seg.start, w2_seg.ys)
    
    aligned_a = Wave(ys1_interp, ts_common, target_framerate)
    aligned_b = Wave(ys2_interp, ts_common, target_framerate)
    return aligned_a, aligned_b

def get_peak_frequency(spectrum):
    # Skip DC component (index 0) to get the dominant AC frequency
    amps = spectrum.amps[1:]
    fs = spectrum.fs[1:]
    if len(amps) == 0:
        return 0.0
    return fs[np.argmax(amps)]

def metric_card(title, value, desc=""):
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-desc">{desc}</div>
        </div>
    """, unsafe_allow_html=True)

# Available signal options list
signal_options = [
    'Chirp', 'SawtoothChirp', 'UncorrelatedUniformNoise', 'BrownianNoise',
    'PinkNoise', 'UncorrelatedGaussianNoise', 'UncorrelatedPoissonNoise',
    'SawtoothSignal', 'SquareSignal', 'CosSignal', 'SinSignal'
]

# Sidebar Navigation
st.sidebar.markdown('<div style="text-align: center; padding: 0.5rem 0;"><span style="font-size: 1.5rem; font-weight: 800; color: #ec4899; font-family: \'Outfit\', sans-serif;">МЕНЮ</span></div>', unsafe_allow_html=True)
page = st.sidebar.radio("Раздел приложения:", ["📊 Анализатор сигналов", "🔄 Сравнение звуков"], key="nav_page")

# ================= PAGE 1: SINGLE SIGNAL ANALYZER & GENERATOR =================
if page == "📊 Анализатор сигналов":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Настройки сигнала")
    source_type = st.sidebar.radio("Источник звука:", ["Сгенерировать сигнал", "Загрузить WAV файл"], key="single_source")
    
    if source_type == "Сгенерировать сигнал":
        sig_type = st.sidebar.selectbox("Тип сигнала:", signal_options, index=1, key="single_sig_type")
        
        # Show end frequency only for chirps
        is_chirp = sig_type in ['Chirp', 'SawtoothChirp']
        
        start_freq = st.sidebar.slider("Частота / Нач. частота (Гц):", 20, 20000, 220, step=10, key="single_start_f")
        
        if is_chirp:
            end_freq = st.sidebar.slider("Конечная частота (Гц):", 20, 20000, 880, step=10, key="single_end_f")
        else:
            end_freq = 0
            
        duration = st.sidebar.slider("Длительность (сек):", 0.1, 5.0, 1.0, step=0.1, key="single_dur")
        framerate = st.sidebar.slider("Частота дискретизации (Гц):", 1000, 48000, 44100, step=1000, key="single_rate")
        
        # Generate wave
        signal_obj = get_signal_object(sig_type, start_freq, end_freq)
        wave_obj = signal_obj.make_wave(duration=duration, framerate=framerate)
        wave_obj.apodize()
        label_title = f"{sig_type} ({start_freq} Гц)" if not is_chirp else f"{sig_type} ({start_freq} -> {end_freq} Гц)"
        
    else:
        uploaded_file = st.sidebar.file_uploader("Загрузите аудиофайл (.wav):", type=["wav"], key="single_upload")
        if uploaded_file is not None:
            wave_obj = load_wav_from_bytes(uploaded_file)
            duration = wave_obj.duration
            framerate = wave_obj.framerate
            label_title = uploaded_file.name
        else:
            st.sidebar.info("Пожалуйста, загрузите WAV файл.")
            wave_obj = None

    # LAG settings
    st.sidebar.markdown("---")
    st.sidebar.subheader("⏱ Настройка лага")
    max_lag_ms_single = st.sidebar.slider(
        "Макс. лаг для графика автокорреляции (мс):",
        min_value=1, max_value=500, value=20, step=1,
        key="single_lag_ms",
        help="Ограничивает диапазон лага на графике автокорреляции"
    )

    # Main Area
    if wave_obj is not None:
        with st.container(border=True):
            st.subheader("Воспроизведение")
            
            # Audio player
            st.write("🔊 **Прослушать сигнал:**")
            audio_bytes = make_audio_bytes(wave_obj)
            st.audio(audio_bytes, format="audio/wav")
        
        with st.container(border=True):
            st.subheader("Аналитические графики (Интерактивные)")
            
            row1_col1, row1_col2 = st.columns(2)
            row2_col1, row2_col2 = st.columns(2)
            row3_col1, row3_col2 = st.columns(2)
            row4_col1, row4_col2 = st.columns(2)
            
            # 1. Segment of wave (first min(0.05, duration) sec)
            with row1_col1:
                segment_duration = min(0.05, duration)
                seg = wave_obj.segment(start=0, duration=segment_duration)
                fig1 = make_plotly_line(
                    seg.ts, seg.ys,
                    f'Фрагмент волны (первые {segment_duration:.3f} сек)',
                    'Время (с)', 'Амплитуда',
                    line_color='#ff4d88'
                )
                st.plotly_chart(fig1, use_container_width=True)
                
            # 2. Full wave
            with row1_col2:
                fig2 = make_plotly_line(
                    wave_obj.ts, wave_obj.ys,
                    f'Полный график волны ({duration:.2f} сек)',
                    'Время (с)', 'Амплитуда',
                    line_color='#ec4899'
                )
                st.plotly_chart(fig2, use_container_width=True)
                
            # 3. Amplitude Spectrum
            spectrum = wave_obj.make_spectrum()
            high_f = framerate / 2
            i_idx = thinkdsp.find_index(high_f, spectrum.fs)
            with row2_col1:
                fig3 = make_plotly_line(
                    spectrum.fs[:i_idx], spectrum.amps[:i_idx],
                    'Спектр амплитуд',
                    'Частота (Гц)', 'Амплитуда',
                    line_color='#db2777'
                )
                st.plotly_chart(fig3, use_container_width=True)
                
            # 4. Power Spectrum
            with row2_col2:
                fig4 = make_plotly_line(
                    spectrum.fs[:i_idx], spectrum.power[:i_idx],
                    'Спектр мощности',
                    'Частота (Гц)', 'Мощность',
                    line_color='#f43f5e'
                )
                st.plotly_chart(fig4, use_container_width=True)
                
            # 5. Integrated Power Spectrum
            with row3_col1:
                integ_spectrum = spectrum.make_integrated_spectrum()
                fig5 = make_plotly_line(
                    integ_spectrum.fs, integ_spectrum.cs,
                    'Интегрированный спектр мощности',
                    'Частота (Гц)', 'Накопленная мощность',
                    line_color='#ff8da1'
                )
                st.plotly_chart(fig5, use_container_width=True)
                
            # 6. Spectrogram
            with row3_col2:
                seg_length = max(2, min(1024, len(wave_obj.ys) // 2))
                sp = wave_obj.make_spectrogram(seg_length=seg_length)
                fig6 = make_plotly_spectrogram(sp, 'Спектрограмма (Время vs Частота)')
                st.plotly_chart(fig6, use_container_width=True)

            # Compute autocorrelation via FFT for full wave length (O(N log N))
            ys_ac = wave_obj.ys
            # Serial correlation — configurable lag
            lag_samples_single = max(1, min(int(round(max_lag_ms_single / 1000.0 * wave_obj.framerate)), len(ys_ac) - 1))
            serial_corr_val = float(np.corrcoef(ys_ac[:-lag_samples_single], ys_ac[lag_samples_single:])[0, 1])
            # FFT-based autocorrelation (full length)
            n_fft = len(ys_ac)
            fft_ac = np.fft.rfft(ys_ac, n=2 * n_fft)
            autocorr_full = np.fft.irfft(fft_ac * np.conj(fft_ac))[:n_fft]
            autocorr_full = autocorr_full / autocorr_full[0]  # Normalize
            lags_ac = np.arange(n_fft) / wave_obj.framerate

            # Serial correlation metric
            st.markdown(
                f"""
                <div class="metric-card" style="text-align:center; margin-bottom:1rem;">
                    <div class="metric-title">Последовательная корреляция (serial_corr, lag={lag_samples_single} отсчётов = {max_lag_ms_single} мс)</div>
                    <div class="metric-value" style="font-size:1.8rem;">{serial_corr_val:.4f}</div>
                    <div class="metric-desc">{"Сильная положительная" if serial_corr_val > 0.7 else "Сильная отрицательная" if serial_corr_val < -0.7 else "Слабая / умеренная"} последовательная зависимость</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # 7. Autocorrelation — full wave length
            with row4_col1:
                fig7 = make_plotly_line(
                    lags_ac, autocorr_full,
                    f'Автокорреляция сигнала (полная длина, {n_fft} отсчётов)',
                    'Лаг (с)', 'Нормализованная корреляция',
                    line_color='#c026d3'
                )
                st.plotly_chart(fig7, use_container_width=True)

            # 8. Autocorrelation — configurable lag zoom
            with row4_col2:
                zoom_ac_sec = min(max_lag_ms_single / 1000.0, lags_ac[-1])
                zoom_ac_idx = np.searchsorted(lags_ac, zoom_ac_sec)
                zoom_ac_idx = max(zoom_ac_idx, 1)
                fig8 = make_plotly_line(
                    lags_ac[:zoom_ac_idx], autocorr_full[:zoom_ac_idx],
                    f'Автокорреляция (первые {max_lag_ms_single} мс)',
                    'Лаг (с)', 'Нормализованная корреляция',
                    line_color='#a21caf'
                )
                st.plotly_chart(fig8, use_container_width=True)
    else:
        st.info("👈 Пожалуйста, выберите/загрузите источник звука в боковой панели.")


# ================= PAGE 2: SOUND COMPARISON =================
elif page == "🔄 Сравнение звуков":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Настройка звуков")
    
    with st.sidebar.expander("🟢 Настройки Звука А", expanded=True):
        source_a = st.radio("Источник для Звука А:", ["Генератор", "WAV файл"], key="source_a")
        
        if source_a == "Генератор":
            sig_type_a = st.selectbox("Тип сигнала А:", signal_options, index=9, key="sig_type_a")
            is_chirp_a = sig_type_a in ['Chirp', 'SawtoothChirp']
            
            freq_a = st.slider("Частота А (Гц):", 20, 20000, 440, step=10, key="freq_a")
            if is_chirp_a:
                end_freq_a = st.slider("Кон. частота А (Гц):", 20, 20000, 880, step=10, key="end_freq_a")
            else:
                end_freq_a = 0
                
            dur_a = st.slider("Длительность А (сек):", 0.1, 5.0, 1.0, step=0.1, key="dur_a")
            rate_a = st.slider("Частота дискр. А (Гц):", 1000, 48000, 44100, step=1000, key="rate_a")
            
            sig_obj_a = get_signal_object(sig_type_a, freq_a, end_freq_a)
            wave_a = sig_obj_a.make_wave(duration=dur_a, framerate=rate_a)
            wave_a.apodize()
            name_a = f"Сигнал A: {sig_type_a} ({freq_a} Гц)" if not is_chirp_a else f"Сигнал A: {sig_type_a} ({freq_a}->{end_freq_a} Гц)"
        else:
            upload_a = st.file_uploader("Загрузите WAV файл А:", type=["wav"], key="upload_a")
            if upload_a is not None:
                wave_a = load_wav_from_bytes(upload_a)
                name_a = f"Файл A: {upload_a.name}"
            else:
                st.info("Загрузите файл для Звука А")
                wave_a = None
                name_a = "Звук А (Ожидание)"
                
    with st.sidebar.expander("🔴 Настройки Звука Б", expanded=True):
        source_b = st.radio("Источник для Звука Б:", ["Генератор", "WAV файл"], key="source_b")
        
        if source_b == "Генератор":
            sig_type_b = st.selectbox("Тип сигнала Б:", signal_options, index=10, key="sig_type_b")
            is_chirp_b = sig_type_b in ['Chirp', 'SawtoothChirp']
            
            freq_b = st.slider("Частота Б (Гц):", 20, 20000, 440, step=10, key="freq_b")
            if is_chirp_b:
                end_freq_b = st.slider("Кон. частота Б (Гц):", 20, 20000, 880, step=10, key="end_freq_b")
            else:
                end_freq_b = 0
                
            dur_b = st.slider("Длительность Б (сек):", 0.1, 5.0, 1.0, step=0.1, key="dur_b")
            rate_b = st.slider("Частота дискр. Б (Гц):", 1000, 48000, 44100, step=1000, key="rate_b")
            
            sig_obj_b = get_signal_object(sig_type_b, freq_b, end_freq_b)
            wave_b = sig_obj_b.make_wave(duration=dur_b, framerate=rate_b)
            wave_b.apodize()
            name_b = f"Сигнал Б: {sig_type_b} ({freq_b} Гц)" if not is_chirp_b else f"Сигнал Б: {sig_type_b} ({freq_b}->{end_freq_b} Гц)"
        else:
            upload_b = st.file_uploader("Загрузите WAV файл Б:", type=["wav"], key="upload_b")
            if upload_b is not None:
                wave_b = load_wav_from_bytes(upload_b)
                name_b = f"Файл Б: {upload_b.name}"
            else:
                st.info("Загрузите файл для Звука Б")
                wave_b = None
                name_b = "Звук Б (Ожидание)"

    # LAG settings
    st.sidebar.markdown("---")
    st.sidebar.subheader("⏱ Настройка лага")
    max_lag_ms_comp = st.sidebar.slider(
        "Макс. лаг для графиков корреляции (мс):",
        min_value=1, max_value=1000, value=50, step=1,
        key="comp_lag_ms",
        help="Ограничивает диапазон лага на графиках взаимной и автокорреляций"
    )
    show_full_crosscorr = st.sidebar.checkbox(
        "Показать полную взаимную корреляцию",
        value=False,
        key="comp_show_full_cc",
        help="Если включено — отображает взаимную корреляцию на всей длине записи"
    )

    # Main Area
    if wave_a is not None and wave_b is not None:
        # Crop both waves to 5 seconds if they exceed it
        if wave_a.duration > 5.0:
            wave_a = wave_a.segment(start=0, duration=5.0)
        if wave_b.duration > 5.0:
            wave_b = wave_b.segment(start=0, duration=5.0)
            
        with st.container(border=True):
            st.subheader("Прослушивание")
            
            # Audio Players Side-by-Side
            col_play_a, col_play_b = st.columns(2)
            with col_play_a:
                st.markdown(f"**🟢 {name_a}**")
                st.audio(make_audio_bytes(wave_a), format="audio/wav")
            with col_play_b:
                st.markdown(f"**🔴 {name_b}**")
                st.audio(make_audio_bytes(wave_b), format="audio/wav")
                
        # Compute spectra for visualizations
        spec_a = wave_a.make_spectrum()
        spec_b = wave_b.make_spectrum()
            
        with st.container(border=True):
            st.subheader("Визуальное сравнение")
            
            # Selection of overlay or side-by-side
            view_mode = st.selectbox(
                "Режим сравнения графиков:",
                options=["Наложение", "Рядом"],
                index=0
            )
            
            # --- Section 1: Waveforms ---
            st.markdown("#### 📈 Сравнение волновых форм")
            
            if view_mode == "Наложение":
                # Overlaid waveforms
                fig_comp_time = make_plotly_overlaid_lines(
                    wave_a.ts, wave_a.ys, "Звук А",
                    wave_b.ts, wave_b.ys, "Звук Б",
                    "Наложение волновых форм во времени",
                    "Время (с)", "Амплитуда",
                    color_a='#ff4d88', color_b='#9333ea'
                )
                st.plotly_chart(fig_comp_time, use_container_width=True)
                
                # Zoomed overlaid waveforms
                zoom_dur = min(0.05, wave_a.duration, wave_b.duration)
                seg_a_z = wave_a.segment(start=0, duration=zoom_dur)
                seg_b_z = wave_b.segment(start=0, duration=zoom_dur)
                fig_comp_time_zoom = make_plotly_overlaid_lines(
                    seg_a_z.ts, seg_a_z.ys, "Звук А (фрагмент)",
                    seg_b_z.ts, seg_b_z.ys, "Звук Б (фрагмент)",
                    "Детальный фрагмент волновых форм (первые 50 мс)",
                    "Время (с)", "Амплитуда",
                    color_a='#ff4d88', color_b='#9333ea',
                    is_dashed_b=True
                )
                st.plotly_chart(fig_comp_time_zoom, use_container_width=True)
            else:
                # Side-by-side waveforms
                col_w_a, col_w_b = st.columns(2)
                with col_w_a:
                    fig_w_a = make_plotly_line(
                        wave_a.ts, wave_a.ys,
                        "Полный график волны (Звук А)",
                        "Время (с)", "Амплитуда",
                        line_color='#ff4d88'
                    )
                    st.plotly_chart(fig_w_a, use_container_width=True)
                    
                    zoom_dur_a = min(0.05, wave_a.duration)
                    seg_a_z = wave_a.segment(start=0, duration=zoom_dur_a)
                    fig_w_a_z = make_plotly_line(
                        seg_a_z.ts, seg_a_z.ys,
                        "Детальный фрагмент волны (Звук А, первые 50 мс)",
                        "Время (с)", "Амплитуда",
                        line_color='#ff4d88'
                    )
                    st.plotly_chart(fig_w_a_z, use_container_width=True)
                    
                with col_w_b:
                    fig_w_b = make_plotly_line(
                        wave_b.ts, wave_b.ys,
                        "Полный график волны (Звук Б)",
                        "Время (с)", "Амплитуда",
                        line_color='#9333ea'
                    )
                    st.plotly_chart(fig_w_b, use_container_width=True)
                    
                    zoom_dur_b = min(0.05, wave_b.duration)
                    seg_b_z = wave_b.segment(start=0, duration=zoom_dur_b)
                    fig_w_b_z = make_plotly_line(
                        seg_b_z.ts, seg_b_z.ys,
                        "Детальный фрагмент волны (Звук Б, первые 50 мс)",
                        "Время (с)", "Амплитуда",
                        line_color='#9333ea'
                    )
                    st.plotly_chart(fig_w_b_z, use_container_width=True)
            
            # --- Section 2: Amplitude Spectra ---
            st.markdown("#### ⚡ Сравнение спектров амплитуд")
            high_cutoff = min(wave_a.framerate / 2, wave_b.framerate / 2)
            i_a = thinkdsp.find_index(high_cutoff, spec_a.fs)
            i_b = thinkdsp.find_index(high_cutoff, spec_b.fs)
            
            if view_mode == "Наложение":
                fig_comp_spec = make_plotly_overlaid_lines(
                    spec_a.fs[:i_a], spec_a.amps[:i_a], "Звук А",
                    spec_b.fs[:i_b], spec_b.amps[:i_b], "Звук Б",
                    f"Наложение спектров амплитуд (до {high_cutoff:.0f} Гц)",
                    "Частота (Гц)", "Амплитуда",
                    color_a='#ff4d88', color_b='#9333ea'
                )
                st.plotly_chart(fig_comp_spec, use_container_width=True)
            else:
                col_s_a, col_s_b = st.columns(2)
                with col_s_a:
                    fig_s_a = make_plotly_line(
                        spec_a.fs[:i_a], spec_a.amps[:i_a],
                        f"Спектр амплитуд (Звук А, до {high_cutoff:.0f} Гц)",
                        "Частота (Гц)", "Амплитуда",
                        line_color='#ff4d88'
                    )
                    st.plotly_chart(fig_s_a, use_container_width=True)
                with col_s_b:
                    fig_s_b = make_plotly_line(
                        spec_b.fs[:i_b], spec_b.amps[:i_b],
                        f"Спектр амплитуд (Звук Б, до {high_cutoff:.0f} Гц)",
                        "Частота (Гц)", "Амплитуда",
                        line_color='#9333ea'
                    )
                    st.plotly_chart(fig_s_b, use_container_width=True)
            
            # --- Section 2.5: Power Spectra ---
            st.markdown("#### 🔋 Сравнение спектров мощности")
            
            if view_mode == "Наложение":
                fig_comp_power = make_plotly_overlaid_lines(
                    spec_a.fs[:i_a], spec_a.power[:i_a], "Звук А",
                    spec_b.fs[:i_b], spec_b.power[:i_b], "Звук Б",
                    f"Наложение спектров мощности (до {high_cutoff:.0f} Гц)",
                    "Частота (Гц)", "Мощность",
                    color_a='#ff4d88', color_b='#9333ea'
                )
                st.plotly_chart(fig_comp_power, use_container_width=True)
            else:
                col_p_a, col_p_b = st.columns(2)
                with col_p_a:
                    fig_p_a = make_plotly_line(
                        spec_a.fs[:i_a], spec_a.power[:i_a],
                        f"Спектр мощности (Звук А, до {high_cutoff:.0f} Гц)",
                        "Частота (Гц)", "Мощность",
                        line_color='#ff4d88'
                    )
                    st.plotly_chart(fig_p_a, use_container_width=True)
                with col_p_b:
                    fig_p_b = make_plotly_line(
                        spec_b.fs[:i_b], spec_b.power[:i_b],
                        f"Спектр мощности (Звук Б, до {high_cutoff:.0f} Гц)",
                        "Частота (Гц)", "Мощность",
                        line_color='#9333ea'
                    )
                    st.plotly_chart(fig_p_b, use_container_width=True)
            
            # --- Section 3: Spectrograms ---
            st.markdown("#### 🔥 Сравнительные спектрограммы")
            
            col_spec_a, col_spec_b = st.columns(2)
            with col_spec_a:
                seg_a = max(2, min(1024, len(wave_a.ys) // 2))
                sp_a = wave_a.make_spectrogram(seg_length=seg_a)
                fig_spec_a = make_plotly_spectrogram(sp_a, "Спектрограмма Звука А (Частота vs Время)")
                st.plotly_chart(fig_spec_a, use_container_width=True)
                
            with col_spec_b:
                seg_b = max(2, min(1024, len(wave_b.ys) // 2))
                sp_b = wave_b.make_spectrogram(seg_length=seg_b)
                fig_spec_b = make_plotly_spectrogram(sp_b, "Спектрограмма Звука Б (Частота vs Время)")
                st.plotly_chart(fig_spec_b, use_container_width=True)

            # --- Section 4: Correlation Analysis ---
            st.markdown("#### 🔗 Корреляционный анализ")

            # Align waves for correlation
            aligned_a, aligned_b = align_waves(wave_a, wave_b)
            ys_a_aligned = aligned_a.ys
            ys_b_aligned = aligned_b.ys
            n_aligned = len(ys_a_aligned)

            # Pearson correlation coefficient
            pearson_r = float(np.corrcoef(ys_a_aligned, ys_b_aligned)[0, 1])

            # Serial correlations — configurable lag
            lag_samples_comp = max(1, min(int(round(max_lag_ms_comp / 1000.0 * aligned_a.framerate)), n_aligned - 1))
            sc_val_a = float(np.corrcoef(ys_a_aligned[:-lag_samples_comp], ys_a_aligned[lag_samples_comp:])[0, 1])
            sc_val_b = float(np.corrcoef(ys_b_aligned[:-lag_samples_comp], ys_b_aligned[lag_samples_comp:])[0, 1])

            # Metrics row
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.markdown(
                    f"""
                    <div class="metric-card" style="text-align:center;">
                        <div class="metric-title">Коэффициент Пирсона (r)</div>
                        <div class="metric-value" style="font-size:1.8rem;">{pearson_r:.4f}</div>
                        <div class="metric-desc">{"Сильная полож." if pearson_r > 0.7 else "Сильная отриц." if pearson_r < -0.7 else "Слабая/умеренная"} корреляция</div>
                    </div>
                    """, unsafe_allow_html=True)
            with mc2:
                st.markdown(
                    f"""
                    <div class="metric-card" style="text-align:center;">
                        <div class="metric-title">serial_corr Звука А (lag={lag_samples_comp} отсч. = {max_lag_ms_comp} мс)</div>
                        <div class="metric-value" style="font-size:1.8rem;">{sc_val_a:.4f}</div>
                        <div class="metric-desc">{"Высокая" if abs(sc_val_a) > 0.7 else "Умеренная" if abs(sc_val_a) > 0.3 else "Слабая"} последовательная зависимость</div>
                    </div>
                    """, unsafe_allow_html=True)
            with mc3:
                st.markdown(
                    f"""
                    <div class="metric-card" style="text-align:center;">
                        <div class="metric-title">serial_corr Звука Б (lag={lag_samples_comp} отсч. = {max_lag_ms_comp} мс)</div>
                        <div class="metric-value" style="font-size:1.8rem;">{sc_val_b:.4f}</div>
                        <div class="metric-desc">{"Высокая" if abs(sc_val_b) > 0.7 else "Умеренная" if abs(sc_val_b) > 0.3 else "Слабая"} последовательная зависимость</div>
                    </div>
                    """, unsafe_allow_html=True)

            col_corr1, col_corr2 = st.columns(2)

            # Cross-correlation via FFT — configurable lag
            with col_corr1:
                n_fft_cc = 2 * n_aligned
                fft_a = np.fft.rfft(ys_a_aligned, n=n_fft_cc)
                fft_b = np.fft.rfft(ys_b_aligned, n=n_fft_cc)
                cross_corr_full = np.fft.irfft(fft_a * np.conj(fft_b))[:n_aligned]
                norm_cc = np.sqrt(np.sum(ys_a_aligned**2) * np.sum(ys_b_aligned**2))
                if norm_cc > 0:
                    cross_corr_full = cross_corr_full / norm_cc
                lags_cc = np.arange(n_aligned) / aligned_a.framerate

                if show_full_crosscorr:
                    cc_x, cc_y = lags_cc, cross_corr_full
                    cc_title = f'Взаимная корреляция (полная длина, {n_aligned} отсчётов)'
                else:
                    zoom_cc_sec = min(max_lag_ms_comp / 1000.0, lags_cc[-1])
                    zoom_cc_idx = max(np.searchsorted(lags_cc, zoom_cc_sec), 1)
                    cc_x, cc_y = lags_cc[:zoom_cc_idx], cross_corr_full[:zoom_cc_idx]
                    cc_title = f'Взаимная корреляция (первые {max_lag_ms_comp} мс)'

                fig_cc = make_plotly_line(
                    cc_x, cc_y,
                    cc_title,
                    'Лаг (с)', 'Нормализованная корреляция',
                    line_color='#7c3aed'
                )
                st.plotly_chart(fig_cc, use_container_width=True)

            # Scatter correlation plot — downsample only for display
            with col_corr2:
                n_scatter = min(n_aligned, 3000)
                idx_scatter = np.linspace(0, n_aligned - 1, n_scatter, dtype=int)
                sc_a_pts = ys_a_aligned[idx_scatter]
                sc_b_pts = ys_b_aligned[idx_scatter]

                fig_scatter = go.Figure()
                fig_scatter.add_trace(go.Scatter(
                    x=sc_a_pts, y=sc_b_pts,
                    mode='markers',
                    marker=dict(color='#7c3aed', size=3, opacity=0.4),
                    name='Точки'
                ))
                m_fit = np.polyfit(sc_a_pts, sc_b_pts, 1)
                x_fit = np.linspace(sc_a_pts.min(), sc_a_pts.max(), 100)
                y_fit = np.polyval(m_fit, x_fit)
                fig_scatter.add_trace(go.Scatter(
                    x=x_fit, y=y_fit,
                    mode='lines',
                    line=dict(color='#ff4d88', width=2, dash='dash'),
                    name=f'Линия тренда (r={pearson_r:.3f})'
                ))
                fig_scatter.update_layout(
                    title=dict(
                        text=f'График корреляции (Звук А vs Звук Б, {n_aligned} отсчётов)',
                        font=dict(size=14, family='Outfit, sans-serif', color='#3a0f28', weight='bold'),
                        x=0.5, xanchor='center'
                    ),
                    xaxis=dict(
                        title=dict(text='Амплитуда Звука А', font=dict(size=11, color='#8b3a62')),
                        tickfont=dict(size=9, color='#8b3a62'),
                        gridcolor='#ffe3e8', linecolor='#ffb3c1'
                    ),
                    yaxis=dict(
                        title=dict(text='Амплитуда Звука Б', font=dict(size=11, color='#8b3a62')),
                        tickfont=dict(size=9, color='#8b3a62'),
                        gridcolor='#ffe3e8', linecolor='#ffb3c1'
                    ),
                    margin=dict(l=45, r=15, t=45, b=40),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='#ffffff',
                    showlegend=True,
                    legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)', bordercolor='#ffb3c1', borderwidth=1)
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

            # --- Section 5: Autocorrelations (full length via FFT) ---
            st.markdown("#### 🔁 Автокорреляции")
            col_ac_a, col_ac_b = st.columns(2)

            for col_ac, wave_xy, name_xy, color_xy in [
                (col_ac_a, wave_a, name_a, '#ff4d88'),
                (col_ac_b, wave_b, name_b, '#9333ea')
            ]:
                with col_ac:
                    ys_xy = wave_xy.ys
                    n_xy = len(ys_xy)
                    fft_xy = np.fft.rfft(ys_xy, n=2 * n_xy)
                    ac_xy = np.fft.irfft(fft_xy * np.conj(fft_xy))[:n_xy]
                    ac_xy = ac_xy / ac_xy[0]  # Normalize
                    lags_xy = np.arange(n_xy) / wave_xy.framerate

                    zoom_ac_comp_sec = min(max_lag_ms_comp / 1000.0, lags_xy[-1])
                    zoom_ac_comp_idx = max(np.searchsorted(lags_xy, zoom_ac_comp_sec), 1)

                    fig_ac_xy = make_plotly_line(
                        lags_xy[:zoom_ac_comp_idx], ac_xy[:zoom_ac_comp_idx],
                        f'Автокорреляция ({name_xy}, первые {max_lag_ms_comp} мс)',
                        'Лаг (с)', 'Нормализованная корреляция',
                        line_color=color_xy
                    )
                    st.plotly_chart(fig_ac_xy, use_container_width=True)
    else:
        st.info("👈 Пожалуйста, настройте/загрузите оба звука в боковой панели.")

