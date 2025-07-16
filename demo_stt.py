#!/usr/bin/env python3
"""
Gradio demo for HGRN ASR with real-time audio recording and transcription.
Allows users to record audio, visualize spectrograms, and get transcriptions.
"""

import gradio as gr
import numpy as np
import torch
import torchaudio
import librosa
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import base64
from PIL import Image
import tempfile
import os
import argparse
from pathlib import Path

# Import our inference components
from infer_stt import ASRInferenceEngine, AudioPreprocessor

SAMPLE_RATE = 16000


class GradioASRDemo:
    """Gradio demo class for ASR with spectrogram visualization"""
    
    def __init__(self, checkpoint_path, device='cpu', beam_size=5):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.beam_size = beam_size
        
        # Initialize inference components
        print("Loading ASR model...")
        self.inference_engine = ASRInferenceEngine(checkpoint_path, device)
        
        # Get num_mfcc from loaded config
        self.num_mfcc = getattr(self.inference_engine.config, 'input_size', 80)
        self.preprocessor = AudioPreprocessor(num_mfcc=self.num_mfcc)
        
        print(f"Demo initialized with {self.num_mfcc} MFCC features")
        print(f"Vocabulary size: {len(self.inference_engine.char_to_idx)}")
    
    def process_audio_and_transcribe(self, audio_input, decoding_mode, beam_size):
        """
        Process recorded audio, generate spectrogram, and transcribe
        
        Args:
            audio_input: Tuple of (sample_rate, audio_array) from Gradio microphone
            decoding_mode: String indicating "Greedy" or "Beam Search"
            beam_size: Integer beam size for beam search decoding
            
        Returns:
            tuple: (spectrogram_image, transcription_text, processing_info)
        """
        try:
            if audio_input is None:
                return None, "No audio recorded", "Please record some audio first."
            
            sample_rate, audio_array = audio_input
            
            # Convert audio to proper format
            if len(audio_array.shape) > 1:
                # Convert stereo to mono
                audio_array = np.mean(audio_array, axis=1)
            
            # Normalize audio
            audio_array = audio_array.astype(np.float32)
            if np.max(np.abs(audio_array)) > 0:
                audio_array = audio_array / np.max(np.abs(audio_array))
            
            # Resample if necessary
            if sample_rate != SAMPLE_RATE:
                audio_tensor = torch.from_numpy(audio_array).unsqueeze(0)
                resampler = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)
                audio_array = resampler(audio_tensor).squeeze().numpy()
            
            # Generate spectrogram visualization
            spectrogram_image = self.create_spectrogram_image(audio_array)
            
            # Save audio to temporary file for inference
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                # Convert to tensor and save
                audio_tensor = torch.from_numpy(audio_array).unsqueeze(0)
                torchaudio.save(tmp_file.name, audio_tensor, SAMPLE_RATE)
                tmp_audio_path = tmp_file.name
            
            try:
                # Perform inference
                features, input_lengths = self.preprocessor.preprocess(tmp_audio_path)
                
                use_beam_search = (decoding_mode == "Beam Search")
                actual_beam_size = beam_size if use_beam_search else 1
                
                transcription, tokens = self.inference_engine.inference(
                    features,
                    input_lengths,
                    use_beam_search=use_beam_search,
                    beam_size=actual_beam_size,
                    max_length=1000,
                    length_penalty=1.0
                )
                
                # Generate processing info
                audio_duration = len(audio_array) / SAMPLE_RATE
                decode_mode = f"Beam Search (size={actual_beam_size})" if use_beam_search else "Greedy"
                processing_info = f"""
**Processing Information:**
- Audio duration: {audio_duration:.2f} seconds
- Sample rate: {SAMPLE_RATE} Hz
- Feature dimensions: {features.shape}
- Number of tokens: {len(tokens)}
- Decoding mode: {decode_mode}
- Device: {self.device}
                """.strip()
                
                # Clean up temp file
                os.unlink(tmp_audio_path)
                
                return spectrogram_image, transcription, processing_info
                
            except Exception as e:
                # Clean up temp file on error
                if os.path.exists(tmp_audio_path):
                    os.unlink(tmp_audio_path)
                raise e
                
        except Exception as e:
            error_msg = f"Error processing audio: {str(e)}"
            return None, error_msg, error_msg
    
    def create_spectrogram_image(self, audio_array):
        """Create spectrogram visualization image"""
        try:
            # Extract MFCC features (same as model input)
            mfcc = librosa.feature.melspectrogram(
                y=audio_array,
                sr=SAMPLE_RATE,
                n_mels=self.preprocessor.n_mels,
                hop_length=self.preprocessor.hop_length,
                win_length=self.preprocessor.win_length,
                n_fft=self.preprocessor.n_fft,
                fmin=self.preprocessor.f_min,
                fmax=self.preprocessor.f_max
            )
            
            # Apply same normalization as model
            mfcc_db = 20 * np.log10(mfcc + 1e-10)
            mfcc_normalized = (mfcc_db - np.mean(mfcc_db, axis=1, keepdims=True)) + 1e-8
            
            # Create plot
            plt.figure(figsize=(12, 4))
            
            # Plot raw spectrogram only
            plt.imshow(mfcc_db, aspect='auto', origin='lower', cmap='viridis')
            plt.colorbar(label='dB')
            plt.title('Mel Spectrogram (Raw)')
            plt.ylabel('Mel Frequency Bins')
            plt.xlabel('Time Frames')
            
            # Add info text
            duration = len(audio_array) / SAMPLE_RATE
            plt.suptitle(f'Audio Duration: {duration:.2f}s | Features: {mfcc.shape[0]} x {mfcc.shape[1]} | Sample Rate: {SAMPLE_RATE}Hz')
            
            plt.tight_layout()
            
            # Convert plot to image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            
            # Convert to PIL Image and make a copy to avoid buffer issues
            image = Image.open(buf)
            image = image.copy()  # Make a copy to ensure data is loaded
            
            plt.close()  # Clean up
            buf.close()
            
            return image
            
        except Exception as e:
            # Return error image
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, f'Error generating spectrogram:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title('Spectrogram Generation Error')
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            image = Image.open(buf)
            image = image.copy()  # Make a copy to ensure data is loaded
            plt.close()
            buf.close()
            
            return image
    
    def create_demo_interface(self):
        """Create the Gradio interface"""
        
        # Custom CSS for better styling
        css = """
        .gradio-container {
            max-width: 1200px !important;
        }
        .output-text {
            font-size: 18px !important;
            font-weight: bold !important;
        }
        .info-text {
            font-family: monospace !important;
        }
        """
        
        with gr.Blocks(css=css, title="HGRN ASR Demo") as demo:
            gr.Markdown("""
            # 🎤 HGRN Automatic Speech Recognition Demo
            
            Record audio using your microphone, visualize the mel spectrogram features, and get real-time transcriptions!
            
            **Instructions:**
            1. Click the microphone button to start recording
            2. Speak clearly into your microphone
            3. Click stop when done
            4. Choose decoding mode (Greedy for speed, Beam Search for quality)
            5. If using Beam Search, adjust beam size for quality/speed trade-off
            6. Click "Transcribe Audio" to get results
            7. View the spectrogram and transcription results
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Audio input
                    audio_input = gr.Audio(
                        sources=["microphone"], 
                        type="numpy",
                        label="🎤 Record Audio",
                        format="wav"
                    )
                    
                    # Decoding mode selection
                    decoding_mode = gr.Radio(
                        choices=["Greedy", "Beam Search"],
                        value="Greedy",
                        label="🔍 Decoding Mode",
                        info="Greedy is faster, Beam Search may be more accurate"
                    )
                    
                    # Beam size slider (only relevant for beam search)
                    beam_size_slider = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1,
                        label="🎯 Beam Size",
                        info="Number of beams for beam search (higher = more thorough but slower)",
                        visible=False
                    )
                    
                    # Process button
                    process_btn = gr.Button(
                        "🔄 Transcribe Audio",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column(scale=2):
                    # Spectrogram output
                    spectrogram_output = gr.Image(
                        label="📊 Mel Spectrogram (Model Input Features)",
                        type="pil"
                    )
            
            with gr.Row():
                with gr.Column():
                    # Transcription output
                    transcription_output = gr.Textbox(
                        label="📝 Transcription",
                        placeholder="Transcription will appear here...",
                        lines=3,
                        max_lines=5,
                        elem_classes=["output-text"]
                    )
                
                with gr.Column():
                    # Processing info
                    info_output = gr.Markdown(
                        label="ℹ️ Processing Information",
                        elem_classes=["info-text"]
                    )
            
            # Examples section
            gr.Markdown("""
            ### 💡 Tips for Best Results:
            - Speak clearly and at a moderate pace
            - Ensure good microphone quality
            - Minimize background noise
            - **Greedy decoding**: Fast and efficient for most cases
            - **Beam search**: More thorough exploration, potentially better quality
            - For beam search, try beam sizes 3-5 for good balance of quality vs speed
            - Shorter audio clips (5-30 seconds) work best
            """)
            
            # Model information
            gr.Markdown(f"""
            ### 🤖 Model Information:
            - **Model Type:** HGRN-based ASR with CTC Loss
            - **Features:** {self.num_mfcc} Mel-frequency coefficients
            - **Vocabulary Size:** {len(self.inference_engine.char_to_idx)} characters
            - **Device:** {self.device}
            - **Checkpoint:** `{Path(self.checkpoint_path).name}`
            """)
            
            # Function to toggle beam size slider visibility
            def toggle_beam_size_visibility(mode):
                return gr.update(visible=(mode == "Beam Search"))
            
            # Set up event handlers
            process_btn.click(
                fn=self.process_audio_and_transcribe,
                inputs=[audio_input, decoding_mode, beam_size_slider],
                outputs=[spectrogram_output, transcription_output, info_output],
                show_progress=True
            )
            
            # Toggle beam size slider visibility based on decoding mode
            decoding_mode.change(
                fn=toggle_beam_size_visibility,
                inputs=[decoding_mode],
                outputs=[beam_size_slider]
            )
            
            # Also process on audio change for immediate feedback
            audio_input.change(
                fn=lambda audio, mode, beam_size: self.process_audio_and_transcribe(audio, mode, beam_size) if audio is not None else (None, "", ""),
                inputs=[audio_input, decoding_mode, beam_size_slider],
                outputs=[spectrogram_output, transcription_output, info_output],
                show_progress=False
            )
        
        return demo


def parse_args():
    parser = argparse.ArgumentParser(description="Launch Gradio demo for HGRN ASR")
    
    parser.add_argument("--checkpoint", type=str, required=False, default="/home/apierro/stt/lt25-flame/checkpoints/hgrn_asr_20250710_113904/best_model_asr.pt",
                       help="Path to trained model checkpoint (.pt file)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use for inference")
    parser.add_argument("--beam_size", type=int, default=5,
                       help="Default beam size for decoding")
    parser.add_argument("--port", type=int, default=7876,
                       help="Port for Gradio server")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Host for Gradio server")
    parser.add_argument("--share", action="store_true",
                       help="Create public shareable link")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set environment variables to help with networking issues
    os.environ["GRADIO_SERVER_NAME"] = args.host
    os.environ["GRADIO_SERVER_PORT"] = str(args.port)
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    
    # # Validate checkpoint
    # if not os.path.exists(args.checkpoint):
    #     print(f"Error: Checkpoint file not found: {args.checkpoint}")

    
    print(f"Starting HGRN ASR Demo...")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"Host: {args.host}:{args.port}")
    
    try:
        # Create demo
        demo_app = GradioASRDemo(
            checkpoint_path=args.checkpoint,
            device=args.device,
            beam_size=args.beam_size,
        )
        
        # Create interface
        demo = demo_app.create_demo_interface()
        
        # Launch with minimal configuration
        print(f"Launching Gradio interface...")
        demo.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
        )
        
    except Exception as e:
        print(f"Error launching demo: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()