import os
import numpy as np
import librosa
import soundfile as sf
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# --- Flask App Initialization ---
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# --- Folder/Path Configuration ---
STATIC_FOLDER = 'static'
UPLOAD_FOLDER = 'uploads'
os.makedirs(STATIC_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
EDITED_AUDIO_PATH = os.path.join(STATIC_FOLDER, "edited_signal.wav")
AUDITION_AUDIO_PATH = os.path.join(STATIC_FOLDER, "audition_signal.wav")
ALLOWED_EXTENSIONS = None  # Allow all audio formats - librosa/ffmpeg handles conversion

# --- STFT Configuration ---
SR = 22050
N_FFT = 2048
HOP_LENGTH = 512

# --- In-Memory State ---
audio_state = {}

# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename  # Allow any file with an extension - librosa will handle format detection

def get_spectrogram_data(audio, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH):
    # ... (same as before)
    S_complex = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    S_magnitude, _ = librosa.magphase(S_complex)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.times_like(S_magnitude, sr=sr, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(S_magnitude, ref=np.max)
    return {"data": S_db.tolist(), "times": times.tolist(), "freqs": freqs.tolist()}

def get_source_stft(layer_to_get):
    # ... (same as before)
    if layer_to_get == 'harmonic' and 'harmonic_stft' in audio_state:
        return audio_state['harmonic_stft']
    elif layer_to_get == 'percussive' and 'percussive_stft' in audio_state:
        return audio_state['percussive_stft']
    else:
        return librosa.stft(audio_state.get('current_audio', np.array([])), n_fft=N_FFT, hop_length=HOP_LENGTH)

# --- Audio Analysis Endpoint ---
@app.route('/ai/suggest', methods=['POST'])
def ai_suggest():
    if 'current_audio' not in audio_state: return jsonify({"error": "Audio not loaded"}), 400
    
    audio = audio_state['current_audio']
    S_complex = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_mag, _ = librosa.magphase(S_complex)
    S_db = librosa.amplitude_to_db(S_mag, ref=np.max)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    
    # --- Comprehensive Audio Analysis ---
    issues_found = []
    suggestions = []
    
    # Calculate mean spectrum (average energy at each frequency)
    mean_spectrum_db = np.mean(S_db, axis=1)
    
    # 1. Check for electrical hum (50/60 Hz and harmonics)
    for base_hum in [50, 60]:
        hum_detected = False
        for harmonic in [1, 2, 3]:  # Check fundamental and first two harmonics
            hum_freq = base_hum * harmonic
            idx = np.argmin(np.abs(freqs - hum_freq))
            # Check if this frequency is significantly louder than neighbors
            neighbor_range = 5
            start_idx = max(0, idx - neighbor_range)
            end_idx = min(len(mean_spectrum_db), idx + neighbor_range)
            neighbors = np.concatenate([mean_spectrum_db[start_idx:idx], mean_spectrum_db[idx+1:end_idx]])
            if len(neighbors) > 0 and mean_spectrum_db[idx] > np.mean(neighbors) + 6:  # 6dB above neighbors
                hum_detected = True
        if hum_detected:
            issues_found.append(f"{base_hum}Hz electrical hum detected")
            suggestions.append({
                "description": f"Remove {base_hum}Hz electrical hum and harmonics",
                "action": {"type": "attenuate", "freq_min": base_hum - 5, "freq_max": base_hum + 5}
            })
    
    # 2. Check for low-frequency rumble (below 80 Hz)
    low_freq_idx = np.where(freqs < 80)[0]
    if len(low_freq_idx) > 0:
        low_freq_energy = np.mean(mean_spectrum_db[low_freq_idx])
        mid_freq_idx = np.where((freqs >= 200) & (freqs <= 2000))[0]
        mid_freq_energy = np.mean(mean_spectrum_db[mid_freq_idx])
        # If low frequencies are within 10dB of mids, there might be rumble
        if low_freq_energy > mid_freq_energy - 10:
            issues_found.append("Potential low-frequency rumble below 80Hz")
            suggestions.append({
                "description": "High-pass filter: Remove rumble below 80Hz (common in recordings without a proper mic stand or with HVAC noise)",
                "action": {"type": "attenuate", "freq_min": 20, "freq_max": 80}
            })
    
    # 3. Check for muddy low-mids (200-400 Hz buildup)
    mud_freq_idx = np.where((freqs >= 200) & (freqs <= 400))[0]
    clarity_freq_idx = np.where((freqs >= 1000) & (freqs <= 4000))[0]
    if len(mud_freq_idx) > 0 and len(clarity_freq_idx) > 0:
        mud_energy = np.mean(mean_spectrum_db[mud_freq_idx])
        clarity_energy = np.mean(mean_spectrum_db[clarity_freq_idx])
        if mud_energy > clarity_energy + 3:  # Mud region louder than clarity region
            issues_found.append("Potential muddiness in 200-400Hz range")
            suggestions.append({
                "description": "Reduce muddiness: The 200-400Hz range is elevated, which can make audio sound boxy or unclear",
                "action": {"type": "attenuate", "freq_min": 200, "freq_max": 400}
            })
    
    # 4. Check for harsh high frequencies / sibilance (4-8 kHz)
    harsh_freq_idx = np.where((freqs >= 4000) & (freqs <= 8000))[0]
    presence_freq_idx = np.where((freqs >= 1000) & (freqs <= 3000))[0]
    if len(harsh_freq_idx) > 0 and len(presence_freq_idx) > 0:
        harsh_energy = np.mean(mean_spectrum_db[harsh_freq_idx])
        presence_energy = np.mean(mean_spectrum_db[presence_freq_idx])
        if harsh_energy > presence_energy - 3:  # Harsh region nearly as loud as presence
            issues_found.append("Potential harshness/sibilance in 4-8kHz range")
            suggestions.append({
                "description": "Reduce harshness: The 4-8kHz range is prominent, which can cause listener fatigue",
                "action": {"type": "attenuate", "freq_min": 4000, "freq_max": 8000}
            })
    
    # 5. Check for resonant peaks (narrow frequency bands that stick out)
    # Smooth the spectrum and look for peaks that deviate significantly
    from scipy.ndimage import uniform_filter1d
    smoothed = uniform_filter1d(mean_spectrum_db, size=20)
    deviation = mean_spectrum_db - smoothed
    peak_threshold = 8  # dB above smoothed average
    
    # Find peaks in the 100Hz - 5kHz range (most problematic for resonances)
    analysis_range = np.where((freqs >= 100) & (freqs <= 5000))[0]
    for idx in analysis_range:
        if deviation[idx] > peak_threshold:
            peak_freq = freqs[idx]
            # Check if we haven't already flagged this frequency range
            already_covered = any(
                s["action"]["freq_min"] <= peak_freq <= s["action"]["freq_max"] 
                for s in suggestions
            )
            if not already_covered:
                issues_found.append(f"Resonant peak detected around {int(peak_freq)}Hz")
                # Create a narrow notch around the peak
                bandwidth = max(20, peak_freq * 0.1)  # 10% bandwidth or 20Hz minimum
                suggestions.append({
                    "description": f"Reduce resonant peak at {int(peak_freq)}Hz: This frequency stands out unnaturally and may ring or cause feedback",
                    "action": {"type": "attenuate", "freq_min": int(peak_freq - bandwidth/2), "freq_max": int(peak_freq + bandwidth/2)}
                })
    
    # 6. Check for broadband noise floor (hiss)
    high_freq_idx = np.where(freqs >= 8000)[0]
    if len(high_freq_idx) > 0:
        high_freq_energy = np.mean(mean_spectrum_db[high_freq_idx])
        if high_freq_energy > -50:  # If high frequencies aren't quiet, might be hiss
            # Check if it's relatively flat (characteristic of noise)
            high_freq_std = np.std(mean_spectrum_db[high_freq_idx])
            if high_freq_std < 5:  # Relatively flat = likely noise, not musical content
                issues_found.append("Potential high-frequency hiss/noise detected")
                suggestions.append({
                    "description": "Reduce high-frequency hiss: Broadband noise detected above 8kHz. Consider using the Denoise feature for better results.",
                    "action": {"type": "attenuate", "freq_min": 8000, "freq_max": 11000}
                })
    
    # If no issues found, return helpful message
    if not suggestions:
        return jsonify([{
            "description": "Audio appears clean! No obvious spectral issues detected. You can still make manual selections to sculpt the sound.",
            "action": {"type": "info", "freq_min": 0, "freq_max": 0}
        }])
    
    # Limit to top 4 most impactful suggestions
    return jsonify(suggestions[:4])


# --- All other endpoints remain the same ---
@app.route('/')
def index(): return render_template('index.html')
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files: return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        audio, sr = librosa.load(filepath, sr=SR)
        audio_state.clear(); audio_state['current_audio'] = audio
        return jsonify({'audio_path': filepath, 'spectrogram': get_spectrogram_data(audio, sr=sr)})
    return jsonify({"error": "File type not allowed"}), 400
@app.route('/generate', methods=['POST'])
def generate_audio_endpoint():
    duration_s = 5
    t = np.linspace(0, duration_s, int(SR * duration_s), endpoint=False)
    tone = 0.8 * np.sin(2 * np.pi * 440 * t); hiss = 0.05 * np.random.randn(len(t))
    audio = (tone + hiss) / np.max(np.abs(tone + hiss)) * 0.9
    gen_path = os.path.join(STATIC_FOLDER, "generated_signal.wav")
    sf.write(gen_path, audio, SR)
    audio_state.clear(); audio_state['current_audio'] = audio
    return jsonify({'audio_path': gen_path, 'spectrogram': get_spectrogram_data(audio)})
@app.route('/denoise/capture', methods=['POST'])
def capture_noise_profile():
    if 'current_audio' not in audio_state: return jsonify({"error": "Audio not loaded"}), 400
    data = request.get_json()
    source_stft = get_source_stft(data.get('layer', 'current_audio'))
    freqs, times = librosa.fft_frequencies(sr=SR, n_fft=N_FFT), librosa.times_like(source_stft, sr=SR, hop_length=HOP_LENGTH)
    min_bin, max_bin = np.argmin(np.abs(freqs - data.get('freq_min'))), np.argmin(np.abs(freqs - data.get('freq_max')))
    start_frame, end_frame = np.argmin(np.abs(times - data.get('time_start'))), np.argmin(np.abs(times - data.get('time_end')))
    noise_stft_region = source_stft[min_bin:max_bin+1, start_frame:end_frame+1]
    audio_state['noise_profile'] = np.mean(np.abs(noise_stft_region), axis=1)
    audio_state['noise_profile_min_bin'] = min_bin
    return jsonify({"status": "Noise profile captured successfully"})
@app.route('/denoise/apply', methods=['POST'])
def apply_denoise():
    if 'current_audio' not in audio_state: return jsonify({"error": "Audio not loaded"}), 400
    if 'noise_profile' not in audio_state: return jsonify({"error": "Noise profile not captured"}), 400
    data = request.get_json()
    reduction_amount = data.get('amount', 1.0)
    full_stft = librosa.stft(audio_state['current_audio'], n_fft=N_FFT, hop_length=HOP_LENGTH)
    full_mag, full_phase = librosa.magphase(full_stft)
    noise_profile = audio_state['noise_profile']
    min_bin = audio_state['noise_profile_min_bin']
    full_noise_profile_template = np.zeros(full_mag.shape[0])
    full_noise_profile_template[min_bin : min_bin + len(noise_profile)] = noise_profile
    noise_mag = np.tile(full_noise_profile_template, (full_mag.shape[1], 1)).T
    denoised_mag = np.maximum(0, full_mag - reduction_amount * noise_mag)
    denoised_stft = denoised_mag * full_phase
    denoised_audio = librosa.istft(denoised_stft, hop_length=HOP_LENGTH)
    audio_state['current_audio'] = denoised_audio
    sf.write(EDITED_AUDIO_PATH, denoised_audio, SR)
    audio_state.pop('harmonic_stft', None); audio_state.pop('percussive_stft', None)
    return jsonify({'audio_path': EDITED_AUDIO_PATH, 'spectrogram': get_spectrogram_data(denoised_audio)})
@app.route('/audition/region', methods=['POST'])
def audition_region_endpoint():
    if 'current_audio' not in audio_state: return jsonify({"error": "Audio not generated yet."}), 400
    data = request.get_json()
    source_stft = get_source_stft(data.get('layer', 'current_audio'))
    freqs, times = librosa.fft_frequencies(sr=SR, n_fft=N_FFT), librosa.times_like(source_stft, sr=SR, hop_length=HOP_LENGTH)
    min_bin, max_bin = np.argmin(np.abs(freqs - data.get('freq_min'))), np.argmin(np.abs(freqs - data.get('freq_max')))
    start_frame, end_frame = np.argmin(np.abs(times - data.get('time_start'))), np.argmin(np.abs(times - data.get('time_end')))
    
    # Only process the selected time region (much faster for large files)
    # Extract just the time slice we need
    selected_stft = np.zeros((source_stft.shape[0], end_frame - start_frame + 1), dtype=source_stft.dtype)
    selected_stft[min_bin:max_bin+1, :] = source_stft[min_bin:max_bin+1, start_frame:end_frame+1]
    
    audition_audio = librosa.istft(selected_stft, hop_length=HOP_LENGTH)
    
    # Check if there's actually audio content
    peak_amplitude = np.max(np.abs(audition_audio)) if len(audition_audio) > 0 else 0
    duration = len(audition_audio) / SR if len(audition_audio) > 0 else 0
    
    sf.write(AUDITION_AUDIO_PATH, audition_audio, SR)
    return jsonify({
        'audition_path': AUDITION_AUDIO_PATH,
        'peak_amplitude': float(peak_amplitude),
        'duration': float(duration),
        'is_silent': bool(peak_amplitude < 0.01)
    })
@app.route('/separate/hpss', methods=['POST'])
def hpss_endpoint():
    if 'current_audio' not in audio_state: return jsonify({"error": "Audio not generated yet."}), 400
    D_complex = librosa.stft(audio_state['current_audio'], n_fft=N_FFT, hop_length=HOP_LENGTH)
    audio_state['harmonic_stft'], audio_state['percussive_stft'] = librosa.decompose.hpss(D_complex, margin=3.0)
    harmonic_audio = librosa.istft(audio_state['harmonic_stft'], hop_length=HOP_LENGTH)
    percussive_audio = librosa.istft(audio_state['percussive_stft'], hop_length=HOP_LENGTH)
    return jsonify({"harmonic_spectrogram": get_spectrogram_data(harmonic_audio), "percussive_spectrogram": get_spectrogram_data(percussive_audio)})
@app.route('/edit/region', methods=['POST'])
def edit_region_endpoint():
    if 'current_audio' not in audio_state: return jsonify({"error": "Audio not generated yet."}), 400
    data = request.get_json()
    source_stft = get_source_stft(data.get('layer', 'current_audio'))
    freqs, times = librosa.fft_frequencies(sr=SR, n_fft=N_FFT), librosa.times_like(source_stft, sr=SR, hop_length=HOP_LENGTH)
    min_bin, max_bin = np.argmin(np.abs(freqs - data.get('freq_min'))), np.argmin(np.abs(freqs - data.get('freq_max')))
    start_frame, end_frame = np.argmin(np.abs(times - data.get('time_start'))), np.argmin(np.abs(times - data.get('time_end')))
    edited_stft = source_stft.copy()
    edited_stft[min_bin:max_bin+1, start_frame:end_frame+1] *= 0.01
    if data.get('layer') == 'harmonic':
        audio_state['harmonic_stft'] = edited_stft
        edited_audio = librosa.istft(edited_stft + audio_state['percussive_stft'], hop_length=HOP_LENGTH)
        display_audio = librosa.istft(edited_stft, hop_length=HOP_LENGTH)
    elif data.get('layer') == 'percussive':
        audio_state['percussive_stft'] = edited_stft
        edited_audio = librosa.istft(audio_state['harmonic_stft'] + edited_stft, hop_length=HOP_LENGTH)
        display_audio = librosa.istft(edited_stft, hop_length=HOP_LENGTH)
    else:
        edited_audio = librosa.istft(edited_stft, hop_length=HOP_LENGTH)
        audio_state.pop('harmonic_stft', None); audio_state.pop('percussive_stft', None)
        display_audio = edited_audio
    audio_state['current_audio'] = edited_audio
    sf.write(EDITED_AUDIO_PATH, edited_audio, SR)
    return jsonify({'audio_path': EDITED_AUDIO_PATH, 'spectrogram': get_spectrogram_data(display_audio)})

# --- Hum Reduction Endpoint ---
@app.route('/process/hum-reduction', methods=['POST'])
def hum_reduction_endpoint():
    if 'current_audio' not in audio_state: return jsonify({"error": "Audio not loaded"}), 400
    data = request.get_json()
    base_freq = data.get('base_freq', 60)  # 50 or 60 Hz
    num_harmonics = data.get('harmonics', 8)  # Number of harmonics to remove
    bandwidth = data.get('bandwidth', 3)  # Hz bandwidth around each harmonic
    reduction = data.get('reduction', 0.01)  # How much to reduce (0.01 = 99% reduction)
    
    source_stft = librosa.stft(audio_state['current_audio'], n_fft=N_FFT, hop_length=HOP_LENGTH)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    edited_stft = source_stft.copy()
    
    removed_bands = []
    for harmonic in range(1, num_harmonics + 1):
        hum_freq = base_freq * harmonic
        if hum_freq > SR / 2:  # Don't exceed Nyquist
            break
        min_bin = np.argmin(np.abs(freqs - (hum_freq - bandwidth)))
        max_bin = np.argmin(np.abs(freqs - (hum_freq + bandwidth)))
        edited_stft[min_bin:max_bin+1, :] *= reduction
        removed_bands.append(f"{hum_freq}Hz")
    
    edited_audio = librosa.istft(edited_stft, hop_length=HOP_LENGTH)
    audio_state['current_audio'] = edited_audio
    audio_state.pop('harmonic_stft', None)
    audio_state.pop('percussive_stft', None)
    sf.write(EDITED_AUDIO_PATH, edited_audio, SR)
    
    return jsonify({
        'audio_path': EDITED_AUDIO_PATH,
        'spectrogram': get_spectrogram_data(edited_audio),
        'removed_bands': removed_bands
    })

# --- Normalization Endpoint ---
@app.route('/process/normalize', methods=['POST'])
def normalize_endpoint():
    if 'current_audio' not in audio_state: return jsonify({"error": "Audio not loaded"}), 400
    data = request.get_json()
    target_db = data.get('target_db', -1.0)  # Target peak level in dB (default -1dB)
    
    audio = audio_state['current_audio']
    current_peak = np.max(np.abs(audio))
    if current_peak == 0:
        return jsonify({"error": "Audio is silent"}), 400
    
    target_linear = 10 ** (target_db / 20)
    gain = target_linear / current_peak
    normalized_audio = audio * gain
    
    # Ensure no clipping
    normalized_audio = np.clip(normalized_audio, -1.0, 1.0)
    
    audio_state['current_audio'] = normalized_audio
    audio_state.pop('harmonic_stft', None)
    audio_state.pop('percussive_stft', None)
    sf.write(EDITED_AUDIO_PATH, normalized_audio, SR)
    
    gain_db = 20 * np.log10(gain) if gain > 0 else 0
    return jsonify({
        'audio_path': EDITED_AUDIO_PATH,
        'spectrogram': get_spectrogram_data(normalized_audio),
        'gain_applied_db': float(gain_db)
    })

# --- Click Repair Endpoint ---
@app.route('/process/click-repair', methods=['POST'])
def click_repair_endpoint():
    if 'current_audio' not in audio_state: return jsonify({"error": "Audio not loaded"}), 400
    data = request.get_json()
    threshold = data.get('threshold', 3.0)  # Standard deviations above mean
    
    audio = audio_state['current_audio'].copy()
    
    # Detect clicks using derivative (sudden changes)
    diff = np.abs(np.diff(audio))
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    click_threshold = mean_diff + threshold * std_diff
    
    # Find click positions
    click_indices = np.where(diff > click_threshold)[0]
    
    # Interpolate over clicks
    clicks_repaired = 0
    window = 5  # Samples to interpolate around each click
    for idx in click_indices:
        start = max(0, idx - window)
        end = min(len(audio), idx + window + 1)
        if end - start > 2:
            # Linear interpolation
            audio[start:end] = np.linspace(audio[start], audio[min(end, len(audio)-1)], end - start)
            clicks_repaired += 1
    
    audio_state['current_audio'] = audio
    audio_state.pop('harmonic_stft', None)
    audio_state.pop('percussive_stft', None)
    sf.write(EDITED_AUDIO_PATH, audio, SR)
    
    return jsonify({
        'audio_path': EDITED_AUDIO_PATH,
        'spectrogram': get_spectrogram_data(audio),
        'clicks_repaired': clicks_repaired
    })

# --- Boost Region Endpoint ---
@app.route('/edit/boost', methods=['POST'])
def boost_region_endpoint():
    if 'current_audio' not in audio_state: return jsonify({"error": "Audio not generated yet."}), 400
    data = request.get_json()
    boost_amount = data.get('boost', 2.0)  # 2x = +6dB boost
    
    source_stft = get_source_stft(data.get('layer', 'current_audio'))
    freqs, times = librosa.fft_frequencies(sr=SR, n_fft=N_FFT), librosa.times_like(source_stft, sr=SR, hop_length=HOP_LENGTH)
    min_bin, max_bin = np.argmin(np.abs(freqs - data.get('freq_min'))), np.argmin(np.abs(freqs - data.get('freq_max')))
    start_frame, end_frame = np.argmin(np.abs(times - data.get('time_start'))), np.argmin(np.abs(times - data.get('time_end')))
    
    edited_stft = source_stft.copy()
    edited_stft[min_bin:max_bin+1, start_frame:end_frame+1] *= boost_amount
    
    edited_audio = librosa.istft(edited_stft, hop_length=HOP_LENGTH)
    # Soft clip to prevent harsh clipping
    edited_audio = np.tanh(edited_audio)
    
    audio_state['current_audio'] = edited_audio
    audio_state.pop('harmonic_stft', None)
    audio_state.pop('percussive_stft', None)
    sf.write(EDITED_AUDIO_PATH, edited_audio, SR)
    
    return jsonify({'audio_path': EDITED_AUDIO_PATH, 'spectrogram': get_spectrogram_data(edited_audio)})

# --- Heal Region Endpoint (Interpolate) ---
@app.route('/edit/heal', methods=['POST'])
def heal_region_endpoint():
    if 'current_audio' not in audio_state: return jsonify({"error": "Audio not generated yet."}), 400
    data = request.get_json()
    
    source_stft = get_source_stft(data.get('layer', 'current_audio'))
    freqs, times = librosa.fft_frequencies(sr=SR, n_fft=N_FFT), librosa.times_like(source_stft, sr=SR, hop_length=HOP_LENGTH)
    min_bin, max_bin = np.argmin(np.abs(freqs - data.get('freq_min'))), np.argmin(np.abs(freqs - data.get('freq_max')))
    start_frame, end_frame = np.argmin(np.abs(times - data.get('time_start'))), np.argmin(np.abs(times - data.get('time_end')))
    
    edited_stft = source_stft.copy()
    
    # Interpolate each frequency bin across time
    for bin_idx in range(min_bin, max_bin + 1):
        left_val = edited_stft[bin_idx, max(0, start_frame - 1)]
        right_val = edited_stft[bin_idx, min(edited_stft.shape[1] - 1, end_frame + 1)]
        num_frames = end_frame - start_frame + 1
        # Linear interpolation in magnitude, preserve phase structure
        left_mag, left_phase = np.abs(left_val), np.angle(left_val)
        right_mag, right_phase = np.abs(right_val), np.angle(right_val)
        for i, frame_idx in enumerate(range(start_frame, end_frame + 1)):
            t = i / max(num_frames - 1, 1)
            interp_mag = left_mag * (1 - t) + right_mag * t
            # Use phase from nearest edge
            interp_phase = left_phase if t < 0.5 else right_phase
            edited_stft[bin_idx, frame_idx] = interp_mag * np.exp(1j * interp_phase)
    
    edited_audio = librosa.istft(edited_stft, hop_length=HOP_LENGTH)
    audio_state['current_audio'] = edited_audio
    audio_state.pop('harmonic_stft', None)
    audio_state.pop('percussive_stft', None)
    sf.write(EDITED_AUDIO_PATH, edited_audio, SR)
    
    return jsonify({'audio_path': EDITED_AUDIO_PATH, 'spectrogram': get_spectrogram_data(edited_audio)})

# --- De-Esser Endpoint ---
@app.route('/process/deesser', methods=['POST'])
def deesser_endpoint():
    if 'current_audio' not in audio_state: return jsonify({"error": "Audio not loaded"}), 400
    data = request.get_json()
    freq_min = data.get('freq_min', 4000)
    freq_max = data.get('freq_max', 9000)
    reduction = data.get('reduction', 0.5)  # 50% reduction (gentler than full attenuation)
    
    source_stft = librosa.stft(audio_state['current_audio'], n_fft=N_FFT, hop_length=HOP_LENGTH)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    
    min_bin = np.argmin(np.abs(freqs - freq_min))
    max_bin = np.argmin(np.abs(freqs - freq_max))
    center_bin = (min_bin + max_bin) // 2
    
    edited_stft = source_stft.copy()
    
    # Apply bell-curve reduction (strongest in center, gentler at edges)
    for bin_idx in range(min_bin, max_bin + 1):
        # Calculate distance from center (0 to 1)
        dist_from_center = abs(bin_idx - center_bin) / max((max_bin - min_bin) / 2, 1)
        # Bell curve: strongest reduction at center
        bin_reduction = reduction + (1 - reduction) * (dist_from_center ** 2)
        edited_stft[bin_idx, :] *= bin_reduction
    
    edited_audio = librosa.istft(edited_stft, hop_length=HOP_LENGTH)
    audio_state['current_audio'] = edited_audio
    audio_state.pop('harmonic_stft', None)
    audio_state.pop('percussive_stft', None)
    sf.write(EDITED_AUDIO_PATH, edited_audio, SR)
    
    return jsonify({
        'audio_path': EDITED_AUDIO_PATH,
        'spectrogram': get_spectrogram_data(edited_audio)
    })

# --- DE-CLIP: Reconstruct clipped peaks ---
@app.route('/process/declip', methods=['POST'])
def declip_endpoint():
    if 'current_audio' not in audio_state: return jsonify({"error": "Audio not loaded"}), 400
    data = request.get_json()
    threshold = data.get('threshold', 0.95)  # Detect clipping above this level
    
    audio = audio_state['current_audio'].copy()
    
    # Find clipped regions (where signal is at or near max)
    clipped_mask = np.abs(audio) >= threshold
    
    # Count clipped samples
    clipped_count = np.sum(clipped_mask)
    
    if clipped_count == 0:
        return jsonify({'audio_path': EDITED_AUDIO_PATH, 'spectrogram': get_spectrogram_data(audio), 'clipped_samples': 0})
    
    # Cubic interpolation to reconstruct clipped regions
    indices = np.arange(len(audio))
    valid_mask = ~clipped_mask
    
    if np.sum(valid_mask) > 3:  # Need at least 4 points for cubic interp
        from scipy.interpolate import interp1d
        # Use cubic interpolation from valid samples
        interp_func = interp1d(indices[valid_mask], audio[valid_mask], kind='cubic', 
                               bounds_error=False, fill_value='extrapolate')
        audio[clipped_mask] = interp_func(indices[clipped_mask])
        # Soft clip to prevent new clipping
        audio = np.tanh(audio * 0.9) / np.tanh(0.9)
    
    audio_state['current_audio'] = audio
    sf.write(EDITED_AUDIO_PATH, audio, SR)
    
    return jsonify({
        'audio_path': EDITED_AUDIO_PATH,
        'spectrogram': get_spectrogram_data(audio),
        'clipped_samples': int(clipped_count)
    })

# --- DE-CRACKLE: Remove vinyl/tape crackle ---
@app.route('/process/decrackle', methods=['POST'])
def decrackle_endpoint():
    if 'current_audio' not in audio_state: return jsonify({"error": "Audio not loaded"}), 400
    data = request.get_json()
    strength = data.get('strength', 0.5)  # 0-1, how aggressive
    
    audio = audio_state['current_audio'].copy()
    
    # Crackle is characterized by very short, isolated spikes
    # Use median filtering to detect and remove them
    from scipy.ndimage import median_filter
    from scipy.signal import medfilt
    
    # Detect crackle: short transients that deviate significantly from local median
    window_size = 5  # Very short window for crackle detection
    median_filtered = medfilt(audio, kernel_size=window_size)
    
    # Find deviations from median
    deviation = np.abs(audio - median_filtered)
    threshold = np.std(deviation) * (3 - strength * 2)  # Lower threshold = more aggressive
    
    # Crackle mask: where deviation is high
    crackle_mask = deviation > threshold
    crackle_count = np.sum(crackle_mask)
    
    # Replace crackle with median filtered values (smooth interpolation)
    audio[crackle_mask] = median_filtered[crackle_mask]
    
    audio_state['current_audio'] = audio
    sf.write(EDITED_AUDIO_PATH, audio, SR)
    
    return jsonify({
        'audio_path': EDITED_AUDIO_PATH,
        'spectrogram': get_spectrogram_data(audio),
        'crackles_removed': int(crackle_count)
    })

# --- LOUDNESS OPTIMIZE (LUFS) ---
@app.route('/process/loudness', methods=['POST'])
def loudness_endpoint():
    if 'current_audio' not in audio_state: return jsonify({"error": "Audio not loaded"}), 400
    data = request.get_json()
    target_lufs = data.get('target_lufs', -14)  # Spotify/YouTube standard
    
    audio = audio_state['current_audio'].copy()
    
    # Simple LUFS approximation using RMS with K-weighting approximation
    # True LUFS requires ITU-R BS.1770-4, but this is a good approximation
    
    # Apply simple K-weighting filter (high-shelf boost, low-shelf cut)
    from scipy.signal import butter, filtfilt
    
    # High shelf boost at 1500Hz (simplified K-weighting)
    b_high, a_high = butter(2, 1500 / (SR/2), btype='high')
    k_weighted = filtfilt(b_high, a_high, audio) * 1.5 + audio * 0.5
    
    # Calculate current LUFS (approximation)
    rms = np.sqrt(np.mean(k_weighted**2))
    current_lufs = 20 * np.log10(rms + 1e-10) - 0.691  # LUFS offset
    
    # Calculate required gain
    gain_db = target_lufs - current_lufs
    gain_linear = 10 ** (gain_db / 20)
    
    # Apply gain with soft limiting to prevent clipping
    audio = audio * gain_linear
    # Soft clip if needed
    if np.max(np.abs(audio)) > 0.99:
        audio = np.tanh(audio * 0.8) / np.tanh(0.8)
    
    audio_state['current_audio'] = audio
    sf.write(EDITED_AUDIO_PATH, audio, SR)
    
    return jsonify({
        'audio_path': EDITED_AUDIO_PATH,
        'spectrogram': get_spectrogram_data(audio),
        'original_lufs': round(current_lufs, 1),
        'target_lufs': target_lufs,
        'gain_applied_db': round(gain_db, 1)
    })

# --- DE-WIND: Remove low-frequency wind rumble ---
@app.route('/process/dewind', methods=['POST'])
def dewind_endpoint():
    if 'current_audio' not in audio_state: return jsonify({"error": "Audio not loaded"}), 400
    data = request.get_json()
    cutoff = data.get('cutoff', 80)  # Hz - wind is typically below 80-100Hz
    strength = data.get('strength', 0.9)  # How much to reduce
    
    audio = audio_state['current_audio'].copy()
    stft = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    
    # Find bins below cutoff
    wind_bins = freqs < cutoff
    
    # Detect wind: look for high energy in low frequencies that varies over time
    # (wind is characterized by fluctuating low-freq energy)
    low_freq_energy = np.abs(stft[wind_bins, :])
    
    # Calculate variance over time for each low bin
    energy_variance = np.var(low_freq_energy, axis=1)
    mean_variance = np.mean(energy_variance)
    
    # Reduce low frequencies proportionally to their variance (more variance = more wind)
    for i, is_wind_bin in enumerate(wind_bins):
        if is_wind_bin and i < len(energy_variance):
            # More reduction for bins with high variance (wind-like)
            reduction = strength * min(1.0, energy_variance[min(i, len(energy_variance)-1)] / (mean_variance + 1e-10))
            reduction = min(reduction, strength)  # Cap at strength
            stft[i, :] *= (1 - reduction)
    
    audio = librosa.istft(stft, hop_length=HOP_LENGTH)
    audio_state['current_audio'] = audio
    sf.write(EDITED_AUDIO_PATH, audio, SR)
    
    return jsonify({
        'audio_path': EDITED_AUDIO_PATH,
        'spectrogram': get_spectrogram_data(audio),
        'cutoff_hz': cutoff
    })

# --- DE-RUSTLE: Remove clothing/lav mic rustle ---
@app.route('/process/derustle', methods=['POST'])
def derustle_endpoint():
    if 'current_audio' not in audio_state: return jsonify({"error": "Audio not loaded"}), 400
    data = request.get_json()
    sensitivity = data.get('sensitivity', 0.5)
    
    audio = audio_state['current_audio'].copy()
    stft = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    
    # Rustle characteristics: broadband noise bursts in 200-2000Hz range
    # with sudden onset and short duration
    rustle_low = 200
    rustle_high = 2000
    rustle_bins = (freqs >= rustle_low) & (freqs <= rustle_high)
    
    # Detect sudden energy increases (rustle onset)
    rustle_energy = np.sum(magnitude[rustle_bins, :], axis=0)
    energy_diff = np.diff(rustle_energy, prepend=rustle_energy[0])
    
    # Find frames with sudden energy increase (potential rustle)
    threshold = np.std(energy_diff) * (3 - sensitivity * 2)
    rustle_frames = energy_diff > threshold
    
    # Expand rustle detection to neighboring frames (rustle has short duration)
    from scipy.ndimage import binary_dilation
    rustle_frames = binary_dilation(rustle_frames, iterations=3)
    
    # Reduce rustle frames in the rustle frequency range
    for i, is_rustle in enumerate(rustle_frames):
        if is_rustle:
            magnitude[rustle_bins, i] *= 0.3  # Reduce by 70%
    
    # Reconstruct
    stft_cleaned = magnitude * np.exp(1j * phase)
    audio = librosa.istft(stft_cleaned, hop_length=HOP_LENGTH)
    
    audio_state['current_audio'] = audio
    sf.write(EDITED_AUDIO_PATH, audio, SR)
    
    return jsonify({
        'audio_path': EDITED_AUDIO_PATH,
        'spectrogram': get_spectrogram_data(audio),
        'rustle_frames_detected': int(np.sum(rustle_frames))
    })

# --- BREATH CONTROL: Reduce breath sounds ---
@app.route('/process/breath-control', methods=['POST'])
def breath_control_endpoint():
    if 'current_audio' not in audio_state: return jsonify({"error": "Audio not loaded"}), 400
    data = request.get_json()
    reduction = data.get('reduction', 0.7)  # How much to reduce breaths (0-1)
    sensitivity = data.get('sensitivity', 0.5)
    
    audio = audio_state['current_audio'].copy()
    stft = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    
    # Breath characteristics: broadband noise centered around 1-4kHz
    # with relatively flat spectrum (unlike voiced sounds with harmonics)
    breath_low = 500
    breath_high = 5000
    breath_bins = (freqs >= breath_low) & (freqs <= breath_high)
    
    # Calculate spectral flatness per frame (breaths are noise-like = high flatness)
    eps = 1e-10
    breath_mag = magnitude[breath_bins, :] + eps
    geometric_mean = np.exp(np.mean(np.log(breath_mag), axis=0))
    arithmetic_mean = np.mean(breath_mag, axis=0)
    spectral_flatness = geometric_mean / (arithmetic_mean + eps)
    
    # Also check for low overall energy (breaths are quieter than vocals)
    frame_energy = np.sum(magnitude, axis=0)
    energy_threshold = np.percentile(frame_energy, 30)  # Below 30th percentile
    
    # Breath frames: high flatness + low energy
    flatness_threshold = 0.3 + (1 - sensitivity) * 0.4
    breath_frames = (spectral_flatness > flatness_threshold) & (frame_energy < energy_threshold)
    
    breath_count = int(np.sum(breath_frames))
    
    # Reduce breath frames
    for i, is_breath in enumerate(breath_frames):
        if is_breath:
            magnitude[:, i] *= (1 - reduction)
    
    stft_cleaned = magnitude * np.exp(1j * phase)
    audio = librosa.istft(stft_cleaned, hop_length=HOP_LENGTH)
    
    audio_state['current_audio'] = audio
    sf.write(EDITED_AUDIO_PATH, audio, SR)
    
    return jsonify({
        'audio_path': EDITED_AUDIO_PATH,
        'spectrogram': get_spectrogram_data(audio),
        'breath_frames_reduced': breath_count
    })

# --- MOUTH DE-CLICK: Remove mouth clicks and lip smacks ---
@app.route('/process/mouth-declick', methods=['POST'])
def mouth_declick_endpoint():
    if 'current_audio' not in audio_state: return jsonify({"error": "Audio not loaded"}), 400
    data = request.get_json()
    sensitivity = data.get('sensitivity', 0.5)
    
    audio = audio_state['current_audio'].copy()
    
    # Mouth clicks are very short (1-10ms), high-frequency transients
    # Different from regular clicks - they occur in specific frequency ranges (2-8kHz)
    
    stft = librosa.stft(audio, n_fft=512, hop_length=128)  # Higher time resolution
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=512)
    
    # Mouth click frequency range
    click_low = 2000
    click_high = 8000
    click_bins = (freqs >= click_low) & (freqs <= click_high)
    
    # Detect sudden transients in click range
    click_energy = np.sum(magnitude[click_bins, :], axis=0)
    
    # Use second derivative to find sharp attacks
    energy_diff = np.diff(click_energy, n=2, prepend=[click_energy[0], click_energy[0]])
    threshold = np.std(energy_diff) * (4 - sensitivity * 3)
    
    click_frames = np.abs(energy_diff) > threshold
    click_count = int(np.sum(click_frames))
    
    # Interpolate click frames from neighbors
    for i, is_click in enumerate(click_frames):
        if is_click and i > 0 and i < magnitude.shape[1] - 1:
            # Interpolate from neighboring frames in click frequency range
            magnitude[click_bins, i] = (magnitude[click_bins, i-1] + magnitude[click_bins, i+1]) / 2
    
    stft_cleaned = magnitude * np.exp(1j * phase)
    audio = librosa.istft(stft_cleaned, hop_length=128)
    
    # Ensure same length
    if len(audio) > len(audio_state['current_audio']):
        audio = audio[:len(audio_state['current_audio'])]
    elif len(audio) < len(audio_state['current_audio']):
        audio = np.pad(audio, (0, len(audio_state['current_audio']) - len(audio)))
    
    audio_state['current_audio'] = audio
    sf.write(EDITED_AUDIO_PATH, audio, SR)
    
    return jsonify({
        'audio_path': EDITED_AUDIO_PATH,
        'spectrogram': get_spectrogram_data(audio),
        'mouth_clicks_removed': click_count
    })

# --- ADVANCED SPECTRAL REPAIR: Context-aware fill with harmonics ---
@app.route('/edit/spectral-repair', methods=['POST'])
def spectral_repair_endpoint():
    if 'current_audio' not in audio_state: return jsonify({"error": "Audio not loaded"}), 400
    data = request.get_json()
    t_start, t_end = data.get('t_start', 0), data.get('t_end', 1)
    f_start, f_end = data.get('f_start', 0), data.get('f_end', 11025)
    
    source_stft = get_source_stft(data.get('layer', 'full'))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    times = librosa.times_like(source_stft, sr=SR, hop_length=HOP_LENGTH)
    
    t_start_idx = np.argmin(np.abs(times - t_start))
    t_end_idx = np.argmin(np.abs(times - t_end))
    f_start_idx = np.argmin(np.abs(freqs - f_start))
    f_end_idx = np.argmin(np.abs(freqs - f_end))
    
    if t_start_idx >= t_end_idx or f_start_idx >= f_end_idx:
        return jsonify({"error": "Invalid selection"}), 400
    
    edited_stft = source_stft.copy()
    magnitude = np.abs(edited_stft)
    phase = np.angle(edited_stft)
    
    # Get context from surrounding region
    context_frames = 10
    left_ctx = max(0, t_start_idx - context_frames)
    right_ctx = min(magnitude.shape[1], t_end_idx + context_frames)
    
    # For each frequency bin in selection, interpolate considering harmonics
    for f_idx in range(f_start_idx, f_end_idx + 1):
        freq = freqs[f_idx]
        
        # Get left and right context magnitudes
        left_mag = np.mean(magnitude[f_idx, left_ctx:t_start_idx]) if t_start_idx > left_ctx else magnitude[f_idx, t_start_idx]
        right_mag = np.mean(magnitude[f_idx, t_end_idx:right_ctx]) if right_ctx > t_end_idx else magnitude[f_idx, t_end_idx]
        
        # Check for harmonic content in context (look for energy at 2x, 3x frequency)
        harmonic_boost = 1.0
        for harmonic in [2, 3]:
            h_freq = freq * harmonic
            h_idx = np.argmin(np.abs(freqs - h_freq))
            if h_idx < len(freqs):
                h_energy = np.mean(magnitude[h_idx, left_ctx:t_start_idx])
                if h_energy > left_mag * 0.1:  # Harmonic is significant
                    harmonic_boost = 1.1  # Slightly boost to preserve harmonic structure
        
        # Smooth interpolation across selection
        for t_idx in range(t_start_idx, t_end_idx + 1):
            alpha = (t_idx - t_start_idx) / max(1, t_end_idx - t_start_idx)
            # Cosine interpolation for smoother transition
            alpha = 0.5 * (1 - np.cos(alpha * np.pi))
            magnitude[f_idx, t_idx] = (left_mag * (1 - alpha) + right_mag * alpha) * harmonic_boost
        
        # Phase: use nearest edge phase with small variation
        left_phase = phase[f_idx, max(0, t_start_idx - 1)]
        right_phase = phase[f_idx, min(phase.shape[1] - 1, t_end_idx + 1)]
        for t_idx in range(t_start_idx, t_end_idx + 1):
            alpha = (t_idx - t_start_idx) / max(1, t_end_idx - t_start_idx)
            # Interpolate phase (handling wrap-around)
            phase_diff = right_phase - left_phase
            if phase_diff > np.pi: phase_diff -= 2 * np.pi
            if phase_diff < -np.pi: phase_diff += 2 * np.pi
            phase[f_idx, t_idx] = left_phase + alpha * phase_diff
    
    edited_stft = magnitude * np.exp(1j * phase)
    edited_audio = librosa.istft(edited_stft, hop_length=HOP_LENGTH)
    
    audio_state['current_audio'] = edited_audio
    audio_state.pop('harmonic_stft', None)
    audio_state.pop('percussive_stft', None)
    sf.write(EDITED_AUDIO_PATH, edited_audio, SR)
    
    return jsonify({
        'audio_path': EDITED_AUDIO_PATH,
        'spectrogram': get_spectrogram_data(edited_audio)
    })

if __name__ == '__main__':
    app.run(port=5001)
