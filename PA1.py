"""
Programming Assignment 1
- Open a wav file
- Read the wav file and add 4096 for each sample with/without saturation
- Write the modified data into wave files 
"""

import wave
import numpy as np


def process_file(input_file, output_file, value, saturation=False):
    with wave.open(input_file, 'rb') as wav_in:
        params = wav_in.getparams()
        # params (nchannels=1, sampwidth=2, framerate=8000, nframes=18800, comptype='NONE', compname='not compressed')
        frames = wav_in.readframes(params.nframes)  # wav파일에서 프레임(18800)을 읽어 바이트(2) 형태로 반환 -> 37600

        # 바이트 데이터를 numpy 배열로 변환
        dtype = np.int16
        audio_data = np.frombuffer(frames, dtype=dtype)  # 부호 있는 16비트 오디오
        # audio1.min(): -6693 / audio1.max(): 23740
        # audio2.min(): -49849 / audio2.max(): 102316

        # 오디오 증폭
        audio_data = (audio_data.astype(np.float32) * 2.5).astype(np.int32)
        modified_audio_data = audio_data + value  # 샘플 값에 4096 더하기

        # saturation 연산
        if saturation:
            modified_audio_data = np.clip(modified_audio_data, -2**15, 2**15-1).astype(np.int16)

        # 수정된 데이터를 바이트 형식으로 변환
        modified_frames = modified_audio_data.astype(dtype).tobytes()

        with wave.open(output_file, 'wb') as wav_out:
            wav_out.setparams(params)
            wav_out.writeframes(modified_frames)
            
            
def save_pcm(wav_file, pcm_file):
    with wave.open(wav_file, 'rb') as wav_in:
        params = wav_in.getparams()
        frames = wav_in.readframes(params.nframes)
        audio_data = np.frombuffer(frames, dtype=np.int16)
        audio_data.tofile(pcm_file)


input_file = './data/a.wav'
output_file_no_saturation = './output/no_saturation.wav'  # saturation 처리 안함
output_file_saturation = './output/saturation.wav'  # saturation 처리
value = 4096
    
# 입력 파일을 saturation 여부에 따라 처리
process_file(input_file, output_file=output_file_no_saturation, value=value, saturation=False)
process_file(input_file, output_file=output_file_saturation, value=value, saturation=True)

# PCM 파일 저장
save_pcm(input_file, './data/a.pcm')
save_pcm(output_file_no_saturation, './output/no_saturation.pcm')
save_pcm(output_file_saturation, './output/saturation.pcm')
