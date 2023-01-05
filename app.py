import os
import whisper
import streamlit as st
from pydub import AudioSegment

st.set_page_config(
    page_title="Whisper based ASR",
    page_icon="musical_note",
    layout="wide",
    initial_sidebar_state="auto",
)

audio_tags = {'comments': 'Converted using pydub!'}


@st.cache(persist=True,allow_output_mutation=False,show_spinner=True,suppress_st_warning=True)
def to_audio(audio_file):
    audio_data = AudioSegment.from_file(audio_file)
    return audio_data

@st.cache(persist=True,allow_output_mutation=False,show_spinner=True,suppress_st_warning=True)
def process_audio(filename, model_type):
    model = whisper.load_model(model_type)
    result = model.transcribe(filename)
    return result["text"]

@st.cache(persist=True,allow_output_mutation=False,show_spinner=True,suppress_st_warning=True)
def save_transcript(txt_file):
    for i in range(len(txt_file["segments"])):
        with open(f"result.txt", "a") as f:
            f.write(txt_file['segments'][i]['text'])
            f.write('\n\n')
            f.close()

st.title("🗣 Automatic Speech Recognition using whisper by OpenAI ✨")
st.info('✨ Supports all popular audio formats - WAV, MP3, MP4, OGG, WMA, AAC, FLAC, FLV ')
uploaded_file = st.file_uploader("Upload audio file", type=["wav","mp3","ogg","wma","aac","flac","mp4","flv"])

audio_file = None

if uploaded_file is not None:
    with st.spinner(f"Processing Audio ... 💫"):
        output_audio_file = to_audio(uploaded_file)
        audio_file = open(output_audio_file, 'rb')
        audio_bytes = audio_file.read()
    print("Opening ",audio_file)
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("Feel free to play your uploaded audio file 🎼")
        st.audio(audio_bytes)
    with col2:
        whisper_model_type = st.radio("Please choose your model type", ('Tiny', 'Base', 'Small', 'Medium', 'Large'))

    if st.button("Generate Transcript"):
        with st.spinner(f"Generating Transcript... 💫"):
            transcript = process_audio(audio_bytes, whisper_model_type.lower())
            output_file = save_transcript(transcript)
            output_file_data = output_file.read()

        if st.download_button(
                            label="Download Transcript 📝",
                            data=output_file_data,
                            file_name='result',
                            mime='text/plain'
                        ):
            st.balloons()
            st.success('✅ Download Successful !!')

else:
    st.warning('⚠ Please upload your audio file 😯')

st.markdown("<br><hr><center><strong>TAISUKE TERAKOSHI</strong></a> with the help of [whisper](https://github.com/openai/whisper) built by [OpenAI](https://github.com/openai) ✨</center><hr>", unsafe_allow_html=True)


