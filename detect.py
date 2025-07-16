import sounddevice as sd
import numpy as np
from collections import deque
from gammatone.gtgram import gtgram

class Recording():#200msずつ録音しバッファに保存
    def __init__(self, samplerate=16000, frame_duration_ms=200):
        self.samplerate = samplerate
        self.frame_duration = frame_duration_ms / 1000  # 秒単位
        self.blocksize = int(self.samplerate * self.frame_duration)
        self.buffer_short = deque(maxlen=4)
        self.buffer_long = deque(maxlen=10)
        self.flag_voice = False

        #InputStreamを初期化
        self.stream = sd.InputStream(#一先ずはモノラル処理
            channels=1,
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            callback=self.set_buffer,
        )
        device_info = sd.query_devices(kind='input')
        print(f"デフォルト入力デバイス: {device_info['name']}")
        print(f"最大入力チャンネル数: {device_info['max_input_channels']}")

    def set_buffer(self,indata,frames,time,status):
        self.buffer_short.append(indata.copy())
        if self.flag_voice:
            self.buffer_long.append(indata.copy())

    def get_buffer(self):
        if self.flag_voice:#音声検知後に長時間バッファに切り替えるための分岐(未実装)
            return list(self.buffer_long)
        else:
            return list(self.buffer_short)

    def start(self):
        self.stream.start()
        print("録音を開始しました")

    def stop(self):
        self.stream.stop()
        self.stream.close()
        print("録音を停止しました")

class Detect_voice():#録音データの処理
    def __init__(self, samplerate=16000, channels=64, f_min=100, f_max=None):
        self.samplerate = samplerate
        self.channels = channels
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else samplerate / 2#f_maxが指定されなければナイキスト周波数を最大に設定
        self.band_center = np.geomspace(self.f_min, self.f_max, self.channels)#filter_hzで分割した周波数帯域毎の代表値
        self.buffer_noise = deque(maxlen=20)

    #音声を聴覚特性に基づいて周波数毎に64分割↓
    def filter_hz(self,signal,window_time=0.025,hop_time=0.010):
        result = gtgram(signal, self.samplerate, window_time, hop_time, self.channels, self.f_min)
        return result

    def filter_level(self,result_hz,threshold_=0.4):#分割した周波数の相対的な強度で認知の有無を判定(発火判定)
        max_ = np.max(result_hz, axis=0, keepdims=True)#簡易実装として最大値の60%
        threshold = threshold_ * max_
        flags = (result_hz >= threshold).astype(np.uint8)
        return flags

    def filter_flags(self,flags):#人の声の周波数帯域に絞ってフラグの本数を保存(未活用)
        flags_sum = []
        for band_idx, hz in enumerate(self.band_center):
            if hz > 265:#肉声の基音が85~255hzである事に準拠
                break
            sum_ = np.sum(flags[band_idx])
            flags_sum.append((sum_,hz))
        return flags_sum

    def noise_cancelling(self,normalized):#持続的なノイズのキャンセル仮実装、もっと長時間分バッファが欲しい、停止中
        if len(self.buffer_noise) == 20:
            noice_count = np.zeros(len(normalized))
            noice_sum = np.zeros(len(normalized))
            for buffer in self.buffer_noise:
                for i in range(len(buffer)):
                    if -0.1 < normalized[i] - buffer[i] < 0.1:
                        noice_count[i] += 1
                        noice_sum[i] += buffer[i]
            for i in range(len(noice_count)):
                if noice_count[i] > 15:
                    normalized[i] -= noice_sum[i] / noice_count[i]
            return normalized
        return normalized

    def detect_harmonics(self,i,normalized,max_,threshold_=0.15):#self.band_center[max_]の帯域周辺に倍音が無いか確認する関数
        if i == 1:#サブハーモニクスの確認
            threshold = self.band_center[max_] / 2
            closest_band = np.argmin(np.abs(self.band_center - threshold))
            strength_fx = normalized[closest_band]
            strength_f0 = normalized[max_]
            lower = (2 - threshold_) * strength_f0
            upper = (2 + threshold_) * strength_f0
            if lower < strength_fx < upper:
                #数値確認用
                #print(normalized)
                #print(f"fx={strength_fx}, f0={strength_f0*2}")
                return True
            else:
                return False
        else:#ハーモニクスの確認
            threshold = self.band_center[max_] * i
            closest_band = np.argmin(np.abs(self.band_center - threshold))
            strength_fx = normalized[closest_band]
            strength_f0 = normalized[max_]
            ratio = 1 / i  # 例: i=2 なら 0.5
            lower = strength_f0 * (ratio - threshold_)
            upper = strength_f0 * (ratio + threshold_)
            if lower < strength_fx < upper:
                #数値確認用
                #print(normalized)
                #print(f"fx={strength_fx}, f0={strength_f0*(1/i)}")
                return True
            else:
                return False


    def detect_voice(self,result_hz,flags_level):#パターンマッチングで肉声の倍音パターンを検索
        mean = np.zeros(result_hz.shape[0])
        for i in range(result_hz.shape[0]):#flags_levelを利用して発火箇所のみ集めて平均を帯域毎に計算
            active_hz = []
            for j in range(len(flags_level[i])):
                if flags_level[i][j] == 1:
                    active_hz.append(result_hz[i][j])
            if len(active_hz) > 0:
                mean[i] = np.mean(active_hz)
            else:
                mean[i] = 0

        if np.max(mean) == 0:
            return False, []

        normalized = mean / np.max(mean)#最大値で正規化、変更予定
        targets = normalized.copy()
        result = []
        while np.max(targets) != 0:#エネルギーの大きい音から順に肉声らしい倍音が無いか判定
            max_ = np.argmax(targets)
            targets[max_] = 0
            result = []
            for i in range(1,6):
                if self.detect_harmonics(i, normalized, max_):
                    result.append(i)
            if len(result) > 3:
                return True, result
        return False, result


def main():
    import time

    R = Recording()
    R.start()

    try:
        D = Detect_voice()
        signal = np.array([], dtype=np.float32)
        while True:
            time.sleep(0.2)
            buffer = R.get_buffer()
            if len(buffer) > 2:
                signal = np.concatenate(buffer, axis=0).reshape(-1)
                result_hz = D.filter_hz(signal)
                flags_level = D.filter_level(result_hz)
                flags_sum = D.filter_flags(flags_level)
                result_detect, result_harmonics = D.detect_voice(result_hz, flags_level)

                #数値確認用
                #for sum_, hz in flags_sum:
                    #print(f"band={hz:.2f}Hz, flag={sum_}")
                #for band_idx, hz in enumerate(D.band_center):
                    #howManyFlag = np.sum(flags_level[band_idx])
                    #print(f"band={hz:.2f}Hz, flag={howManyFlag}")

                if result_detect:
                    print(f"肉声を検知しています: {result_harmonics}")
                else:
                    print(f"肉声が検知できません: {result_harmonics}")
            else:
                print("音源が不足しています")

    except KeyboardInterrupt:
        print("停止します")
    finally:
        R.stop()

if __name__ == "__main__":
    main()
