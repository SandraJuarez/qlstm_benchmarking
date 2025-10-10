# ========= MemoryMonitor: GPU/CPU profiling simple (NVML o nvidia-smi) =========
import threading, time, shutil, subprocess, os, psutil

class MemoryMonitor:
    def __init__(self, gpu_index: int = 0, interval_sec: float = 0.2):
        self.gpu_index = gpu_index
        self.interval = interval_sec
        self.samples = []  # (t, gpu_mem_MiB, cpu_mem_MiB)
        self.marks = []    # [(label, idx)]
        self._stop = threading.Event()
        self._thr = None
        self._use_nvml = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = self._nvml.nvmlDeviceGetHandleByIndex(gpu_index)
            self._use_nvml = True
        except Exception:
            self._nvml = None
            self._handle = None
            # fallback a nvidia-smi
            if not shutil.which("nvidia-smi"):
                print("⚠️ No NVML ni nvidia-smi: solo CPU RAM.")
        self._t0 = None

    def _read_gpu_mem_mib(self):
        if self._use_nvml:
            info = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
            return info.used / (1024**2)  # MiB
        # fallback nvidia-smi
        if shutil.which("nvidia-smi"):
            try:
                out = subprocess.check_output([
                    "nvidia-smi",
                    f"--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                    "-i", str(self.gpu_index)
                ], stderr=subprocess.DEVNULL).decode().strip().splitlines()[0]
                return float(out)
            except Exception:
                return None
        return None

    def _read_cpu_mem_mib(self):
        try:
            return psutil.Process(os.getpid()).memory_info().rss / (1024**2)
        except Exception:
            return None

    def _loop(self):
        self._t0 = time.time()
        while not self._stop.is_set():
            t = time.time() - self._t0
            gpu = self._read_gpu_mem_mib()
            cpu = self._read_cpu_mem_mib()
            self.samples.append((t, gpu, cpu))
            time.sleep(self.interval)

    def start(self):
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()

    def stop(self):
        self._stop.set()
        if self._thr:
            self._thr.join()

    def mark(self, label: str):
        self.marks.append((label, len(self.samples)))

    def segment_peaks(self):
        """Devuelve dict {segment_label: {'gpu_peak':..., 'cpu_peak':..., 't_start':..., 't_end':...}}"""
        segs = {}
        idxs = [idx for _, idx in self.marks] + [len(self.samples)]
        labels = [lbl for lbl, _ in self.marks]
        for k in range(len(labels)):
            a, b = idxs[k], idxs[k+1]
            if a >= b: 
                continue
            seg = self.samples[a:b]
            gpu_vals = [s[1] for s in seg if s[1] is not None]
            cpu_vals = [s[2] for s in seg if s[2] is not None]
            t_start = self.samples[a][0]
            t_end   = self.samples[b-1][0]
            segs[labels[k]] = {
                "gpu_peak_mib": max(gpu_vals) if gpu_vals else None,
                "cpu_peak_mib": max(cpu_vals) if cpu_vals else None,
                "t_start": t_start,
                "t_end": t_end
            }
        return segs
