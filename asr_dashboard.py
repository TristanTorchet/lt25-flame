import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import numpy as np
import base64
from preprocess_ctc import create_asr_dataloaders
import soundfile as sf
import tempfile
import os

# Initialize Dash app
app = dash.Dash(__name__)

# Global variables to store dataset
dataset_cache = {}
current_data = None

def load_dataset(max_samples=100, num_mfcc=80):
    """Load the ASR dataset"""
    global dataset_cache, current_data
    
    if 'data' not in dataset_cache:
        print("Loading ASR dataset...")
        train_loader, _, _, char_to_idx, idx_to_char = create_asr_dataloaders(
            batch_size=1,  # Single samples for visualization
            max_samples=max_samples,
            use_ctc=True,
            num_mfcc=num_mfcc,
            cache_dir="/export/work/apierro/datasets/cache"
        )
        
        # We need to access the raw dataset to get the audio data
        from preprocess_ctc import LibriSpeechASRDataset
        raw_dataset = LibriSpeechASRDataset(
            split="train.100",
            max_samples=max_samples,
            num_mfcc=num_mfcc,
            cache_dir="/export/work/apierro/datasets/cache"
        )
        
        # Extract all samples for visualization, including raw audio
        samples = []
        for batch in train_loader:
            # Get the sample ID from the batch
            sample_id = batch['ids'][0]
            
            # Find the corresponding raw sample by ID
            raw_sample = None
            for raw_s in raw_dataset.samples:
                if raw_s['id'] == sample_id:
                    raw_sample = raw_s
                    break
            
            sample = {
                'features': batch['features'][0],  # Remove batch dimension
                'text': batch['texts'][0],
                'text_tokens': batch['targets'][:batch['target_lengths'][0]],
                'speaker_id': batch['speaker_ids'][0],
                'id': batch['ids'][0],
                'audio': raw_sample['audio'] if raw_sample else None  # Add raw audio data
            }
            samples.append(sample)
        
        dataset_cache['data'] = samples
        dataset_cache['char_to_idx'] = char_to_idx
        dataset_cache['idx_to_char'] = idx_to_char
        print(f"Loaded {len(samples)} samples with audio data")
    
    current_data = dataset_cache['data']
    return dataset_cache['data'], dataset_cache['char_to_idx'], dataset_cache['idx_to_char']

def create_mfcc_plot(features, sample_id):
    """Create MFCC spectrogram plot"""
    # Convert to numpy for plotting
    mfcc = features.numpy()
    
    fig = go.Figure(data=go.Heatmap(
        z=mfcc,
        colorscale='Viridis',
        showscale=True,
        hovertemplate='Time: %{x}<br>MFCC: %{y}<br>Value: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'MFCC Spectrogram - Sample {sample_id}',
        xaxis_title='Time Frames',
        yaxis_title='MFCC Coefficients',
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

def generate_audio_data(sample_idx):
    """Generate audio data for playback using real audio from LibriSpeech"""
    if current_data and sample_idx < len(current_data):
        sample = current_data[sample_idx]
        
        # Check if we have real audio data
        if 'audio' in sample:
            audio_array = sample['audio']
            sample_rate = 16000  # LibriSpeech sample rate
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            sf.write(temp_file.name, audio_array, sample_rate)
            
            # Read back and encode for HTML audio
            with open(temp_file.name, 'rb') as f:
                audio_data = f.read()
            
            # Clean up
            os.unlink(temp_file.name)
            
            # Encode for HTML
            encoded_audio = base64.b64encode(audio_data).decode()
            return f"data:audio/wav;base64,{encoded_audio}"
        else:
            # Fallback to synthetic audio if no real audio available
            text = sample['text']
            duration = max(1.0, len(text) * 0.1)  # 0.1 seconds per character
            sample_rate = 16000
            t = np.linspace(0, duration, int(sample_rate * duration))
            freq = 440 + len(text) * 10  # Base frequency varies with text length
            audio = np.sin(2 * np.pi * freq * t) * 0.3
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            sf.write(temp_file.name, audio, sample_rate)
            
            # Read back and encode for HTML audio
            with open(temp_file.name, 'rb') as f:
                audio_data = f.read()
            
            # Clean up
            os.unlink(temp_file.name)
            
            # Encode for HTML
            encoded_audio = base64.b64encode(audio_data).decode()
            return f"data:audio/wav;base64,{encoded_audio}"
    
    return None

# Load initial dataset
samples, char_to_idx, idx_to_char = load_dataset(max_samples=50)

# Dashboard layout
app.layout = html.Div([
    html.H1("ASR Dataset Visualization Dashboard", 
            style={'textAlign': 'center', 'marginBottom': 30}),
    
    html.Div([
        html.Div([
            html.Label("Select Sample:"),
            dcc.Dropdown(
                id='sample-dropdown',
                options=[{'label': f'Sample {i}: {sample["text"][:50]}...', 'value': i} 
                        for i, sample in enumerate(samples)],
                value=0,
                style={'marginBottom': 20}
            ),
            
            html.Label("Dataset Info:"),
            html.Div(id='dataset-info', style={'marginBottom': 20}),
            
            html.Label("Sample Details:"),
            html.Div(id='sample-details', style={'marginBottom': 20}),
            
            html.Label("Audio Playback:"),
            html.Div(id='audio-player', style={'marginBottom': 20}),
            
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': 20}),
        
        html.Div([
            dcc.Graph(id='mfcc-plot'),
            
            html.Div([
                html.H4("Text Transcription:"),
                html.Div(id='text-display', style={
                    'fontSize': 16, 
                    'padding': 10, 
                    'border': '1px solid #ccc',
                    'borderRadius': 5,
                    'backgroundColor': '#f9f9f9'
                })
            ], style={'marginTop': 20})
            
        ], style={'width': '65%', 'display': 'inline-block', 'padding': 20})
    ]),
    
    html.Div([
        html.H4("Character Tokenization:"),
        html.Div(id='token-display', style={
            'fontSize': 14,
            'padding': 10,
            'border': '1px solid #ccc',
            'borderRadius': 5,
            'backgroundColor': '#f0f0f0',
            'fontFamily': 'monospace'
        })
    ], style={'margin': 20})
])

@app.callback(
    [Output('mfcc-plot', 'figure'),
     Output('text-display', 'children'),
     Output('sample-details', 'children'),
     Output('token-display', 'children'),
     Output('audio-player', 'children')],
    [Input('sample-dropdown', 'value')]
)
def update_visualization(sample_idx):
    if sample_idx is None or sample_idx >= len(samples):
        return {}, "No sample selected", "", "", ""
    
    sample = samples[sample_idx]
    
    # Create MFCC plot
    mfcc_fig = create_mfcc_plot(sample['features'], sample['id'])
    
    # Sample details
    details_items = [
        html.P(f"Sample ID: {sample['id']}"),
        html.P(f"Speaker ID: {sample['speaker_id']}"),
        html.P(f"MFCC Shape: {sample['features'].shape}"),
        html.P(f"Text Length: {len(sample['text'])} characters"),
        html.P(f"Token Length: {len(sample['text_tokens'])} tokens")
    ]
    
    # Add audio information if available
    if 'audio' in sample:
        audio_array = sample['audio']
        sample_rate = 16000
        audio_duration = len(audio_array) / sample_rate
        audio_rms = np.sqrt(np.mean(np.array(audio_array)**2))
        details_items.extend([
            html.P(f"Audio Duration: {audio_duration:.2f} seconds"),
            html.P(f"Audio Samples: {len(audio_array):,}"),
            html.P(f"Audio RMS: {audio_rms:.6f}"),
            html.P("Audio Source: LibriSpeech (Real)", style={'color': 'green', 'fontWeight': 'bold'})
        ])
    else:
        details_items.append(
            html.P("Audio Source: Synthetic", style={'color': 'orange', 'fontWeight': 'bold'})
        )
    
    details = html.Div(details_items)
    
    # Token display
    token_info = []
    for token_id in sample['text_tokens']:
        char = idx_to_char[token_id.item()]
        token_info.append(f"{char}({token_id.item()})")
    
    token_display = " ".join(token_info)
    
    # Generate audio player
    audio_src = generate_audio_data(sample_idx)
    if audio_src:
        audio_player = html.Audio(
            src=audio_src,
            controls=True,
            style={'width': '100%'}
        )
    else:
        audio_player = html.P("Audio not available")
    
    return mfcc_fig, sample['text'], details, token_display, audio_player

@app.callback(
    Output('dataset-info', 'children'),
    [Input('sample-dropdown', 'value')]
)
def update_dataset_info(_):
    info = html.Div([
        html.P(f"Total Samples: {len(samples)}"),
        html.P(f"Vocabulary Size: {len(char_to_idx)}"),
        html.P(f"MFCC Features: {samples[0]['features'].shape[0]}"),
        html.P(f"Blank Token ID: {char_to_idx['<blank>']}")
    ])
    return info

if __name__ == '__main__':
    # Run the app accessible from remote machines
    app.run(debug=True, host='0.0.0.0', port=8050)