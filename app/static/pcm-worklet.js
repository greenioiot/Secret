// Downsamples microphone audio to 16 kHz mono Int16 and posts 20 ms frames to the main thread.
class PCMDownsampler extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.acc = 0;
    this.accN = 0;
    this.pos = 0;
    this.out = new Int16Array(320);
    this.outN = 0;
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      // box-filter average over each output sample period (cheap anti-aliasing)
      this.acc += ch[i];
      this.accN++;
      this.pos += 1;
      if (this.pos >= this.ratio) {
        this.pos -= this.ratio;
        const v = Math.max(-1, Math.min(1, this.acc / this.accN));
        this.acc = 0;
        this.accN = 0;
        this.out[this.outN++] = v < 0 ? v * 0x8000 : v * 0x7fff;
        if (this.outN === this.out.length) {
          this.port.postMessage(this.out.buffer.slice(0));
          this.outN = 0;
        }
      }
    }
    return true;
  }
}
registerProcessor("pcm-downsampler", PCMDownsampler);
