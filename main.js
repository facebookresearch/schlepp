/* Copyright (c) Meta Platforms, Inc. and affiliates. */
// Schlepp project page: inline video playback, the modality/camera viewer,
// the download picker, and copy buttons for code blocks.

(function () {
  // ─── videos never go full screen on phones ───────────────────────────
  // playsinline is set in the markup; also set the property, and retry a
  // blocked autoplay (e.g. iOS Low Power Mode) on the first touch or click.
  document.querySelectorAll("video").forEach((v) => {
    v.playsInline = true;
    if (v.autoplay) {
      v.play().catch(() => {
        const start = () => v.play().catch(() => {});
        document.addEventListener("pointerdown", start, { once: true });
      });
    }
  });
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reducedMotion) {
    document.querySelectorAll("video.teaser").forEach((v) => v.pause());
  }

  // Pause a looping video while it is scrolled out of view, and play it again when it is back
  // in view or the window becomes visible or focused (browsers pause silent video in hidden
  // windows and don't always resume it).
  function playWhenVisible(box, getVideo) {
    let inView = false;
    const play = () => {
      const video = getVideo();
      if (inView && !document.hidden && video.currentSrc) video.play().catch(() => {});
    };
    new IntersectionObserver(([entry]) => {
      inView = entry.isIntersecting;
      if (inView) play(); else getVideo().pause();
    }).observe(box);
    document.addEventListener("visibilitychange", play);
    window.addEventListener("focus", play);
  }
  const teaser = document.querySelector("video.teaser");
  if (teaser && !reducedMotion) playWhenVisible(teaser, () => teaser);

  // ─── segmented controls: slide the indicator under the active button ──
  function placeIndicator(track) {
    const active = track.querySelector("button.active");
    const ind = track.querySelector(".seg-indicator");
    if (!active || !ind) return;
    ind.style.width = `${active.offsetWidth}px`;
    ind.style.transform = `translateX(${active.offsetLeft}px)`;
    // On narrow screens the track scrolls; keep the active button in view.
    const pad = 24;
    if (active.offsetLeft < track.scrollLeft + pad || active.offsetLeft + active.offsetWidth > track.scrollLeft + track.clientWidth - pad) {
      track.scrollTo({ left: active.offsetLeft - (track.clientWidth - active.offsetWidth) / 2, behavior: "smooth" });
    }
  }
  // Arrows beside a track (shown only when it overflows) select the previous / next option.
  function updateArrows(track) {
    const wrap = track.parentElement;
    const buttons = [...track.querySelectorAll("button")];
    const i = buttons.findIndex((b) => b.classList.contains("active"));
    wrap.classList.toggle("overflowing", track.scrollWidth > track.clientWidth + 1);
    wrap.querySelector(".prev").disabled = i <= 0;
    wrap.querySelector(".next").disabled = i >= buttons.length - 1;
  }
  const tracks = document.querySelectorAll(".segmented");
  tracks.forEach((track) => {
    const wrap = track.parentElement;
    const step = (dir) => {
      const buttons = [...track.querySelectorAll("button")];
      const i = buttons.findIndex((b) => b.classList.contains("active"));
      const target = buttons[i + dir];
      if (target) target.click();
    };
    wrap.querySelector(".prev").addEventListener("click", () => step(-1));
    wrap.querySelector(".next").addEventListener("click", () => step(1));
  });
  const placeAll = () => tracks.forEach((t) => { updateArrows(t); placeIndicator(t); });
  placeAll();
  window.addEventListener("resize", placeAll);
  document.fonts && document.fonts.ready.then(placeAll);

  function select(track, btn) {
    track.querySelectorAll("button").forEach((b) => {
      b.classList.toggle("active", b === btn);
      b.setAttribute("aria-selected", b === btn);
    });
    placeIndicator(track);
    updateArrows(track);
  }

  // ─── viewer: modality + camera ───────────────────────────────────────
  const viewer = document.querySelector(".viewer");
  if (viewer) initViewer(viewer);

  function initViewer(viewer) {
    const label = viewer.querySelector("#viewer-label");
    const caption = viewer.querySelector("#viewer-caption");
    const stage = viewer.querySelector(".viewer-stage");
    let video = viewer.querySelector("#viewer-video");
    playWhenVisible(stage, () => video);

    const modalityCopy = {
      rgb: { label: "RGB", text: "Rendered colour frames: 8-bit, 30 fps." },
      depth: { label: "Depth", text: "Metric depth per pixel, in metres. Invalid pixels hold 0." },
      flow_fwd: { label: "Forward flow", text: "Pixel displacement (du, dv) from each frame to the next. Hue shows direction, saturation shows magnitude." },
      flow_bwd: { label: "Backward flow", text: "Pixel displacement (du, dv) from each frame to the previous one, in the same colour coding." },
      segmentation: { label: "Segmentation", text: "One label per pixel, drawn as flat colours. A per-sequence dictionary maps label indices to category names." },
      point_tracks: { label: "Point tracks", text: "About 20K 3D point tracks per sequence, drawn with trails and coloured by category: body, cloth, carried object, scene. Each camera has its own visible and in_frustum masks." },
      obbs: { label: "3D OBBs", text: "Oriented 3D bounding boxes for every object in the scene, drawn as wireframes. Boxes of carried objects move with them." },
    };

    const camCopy = {
      static: "1920 × 1080, pinhole",
      body_follow: "1920 × 1080, pinhole",
      object_orbit: "1920 × 1080, pinhole",
      aria_rgb: "2016 × 1512, KB4 fisheye",
      aria_slamL: "512 × 512, KB4 fisheye",
      aria_slamR: "512 × 512, KB4 fisheye",
    };

    let activeModality = "rgb";
    let activeCam = "body_follow";
    let tourFiles = null;

    // Web encodes (H.264, ≤960px wide) in assets/tour_web; the full-quality originals stay in
    // assets/tour. H.264 plays reliably everywhere (Safari's WebM playback sometimes froze).
    const clipUrl = (cam, modality) => {
      const entry = tourFiles && tourFiles[cam] && tourFiles[cam][modality];
      return entry ? `assets/tour_web/${entry.mp4}` : null;
    };

    function render() {
      const m = modalityCopy[activeModality];
      label.textContent = `${activeCam} · ${m.label}`;
      caption.textContent = `${m.text} Camera: ${activeCam}, ${camCopy[activeCam]}.`;

      // Each clip gets a fresh <video>: Safari keeps stale frame geometry when one element
      // switches between clips of different sizes, and draws the new clip too small.
      const url = clipUrl(activeCam, activeModality);
      if (!url || video.getAttribute("src") === url) return;
      const resumeAt = video.currentTime || 0;
      const next = video.cloneNode();
      next.src = url;
      next.addEventListener("loadedmetadata", () => {
        next.currentTime = resumeAt % next.duration || 0;
        next.play().catch(() => {});
      }, { once: true });
      video.replaceWith(next);
      video = next;
    }

    fetch("assets/tour_web/_layout.json")
      .then((res) => res.json())
      .then((layout) => {
        tourFiles = layout.files;
        render();
      })
      .catch(() => render());

    viewer.querySelectorAll("[data-modality]").forEach((btn) => btn.addEventListener("click", () => {
      select(btn.parentElement, btn);
      activeModality = btn.dataset.modality;
      render();
    }));
    viewer.querySelectorAll("[data-cam]").forEach((btn) => btn.addEventListener("click", () => {
      select(btn.parentElement, btn);
      activeCam = btn.dataset.cam;
      render();
    }));

    // ─── viewer: hover-prefetch ────────────────────────────────────────
    // Warm the HTTP cache for the clip the next click would load, but only
    // when the cursor lingers (≥120ms) and only on real pointers — touch
    // taps fire mouseenter and would prefetch wastefully.
    const prefetched = new Set();
    const finePointer = window.matchMedia("(hover: hover) and (pointer: fine)").matches;

    function prefetchClip(cam, modality) {
      const url = clipUrl(cam, modality);
      if (!url || prefetched.has(url)) return;
      prefetched.add(url);
      fetch(url, { credentials: "omit" }).catch(() => prefetched.delete(url));
    }

    if (finePointer) {
      let hoverTimer = null;
      const arm = (cam, modality) => {
        clearTimeout(hoverTimer);
        hoverTimer = setTimeout(() => prefetchClip(cam, modality), 120);
      };
      const disarm = () => clearTimeout(hoverTimer);
      const watch = (btn, target) => {
        btn.addEventListener("mouseenter", () => arm(...target()));
        btn.addEventListener("focus", () => arm(...target()));
        btn.addEventListener("mouseleave", disarm);
        btn.addEventListener("blur", disarm);
      };
      viewer.querySelectorAll("[data-modality]").forEach((btn) => watch(btn, () => [activeCam, btn.dataset.modality]));
      viewer.querySelectorAll("[data-cam]").forEach((btn) => watch(btn, () => [btn.dataset.cam, activeModality]));
    }

    render();

  }

  // ─── download picker: rewrite the hf --include list ──────────────────
  const picker = document.querySelector(".picker");
  const downloadCode = document.getElementById("download-code");
  if (picker && downloadCode) {
    const boxes = (list) => [...picker.querySelectorAll(`[data-list="${list}"] input:not(:disabled)`)];
    const picked = (list) => boxes(list).filter((b) => b.checked).map((b) => b.value);
    const total = (list) => boxes(list).length;
    const tok = (cls, text) => `<span class="tok-${cls}">${text}</span>`;

    function patterns() {
      const cams = picked("cams"), mods = picked("mods"), out = [];
      if (cams.length === total("cams")) {
        // Every camera: one pattern per modality covers them all.
        mods.forEach((m) => out.push(`*_${m}.tar`));
      } else {
        for (const c of cams) {
          if (mods.length === total("mods")) out.push(`*_${c}_*.tar`);
          else mods.forEach((m) => out.push(`*_${c}_${m}.tar`));
        }
      }
      picked("seq").forEach((m) => out.push(`*_${m}.tar`));
      return [...out, "*_metadata.tar", "*.parquet"];
    }

    function update() {
      const pats = patterns();
      const include = pats
        .map((p, i) => (i ? "              " : "") + tok("str", `"${p}"`))
        .join(" \\\n");
      downloadCode.innerHTML =
        `${tok("fn", "hf")} download facebook/schlepp \\\n` +
        `    ${tok("kw", "--repo-type")} dataset \\\n` +
        `    ${tok("kw", "--local-dir")} /data/schlepp \\\n` +
        `    ${tok("kw", "--include")} ${include}\n\n` +
        `${tok("fn", "git")} clone https://github.com/facebookresearch/schlepp.git\n` +
        `${tok("fn", "pip")} install ${tok("kw", "-e")} schlepp\n\n` +
        `${tok("fn", "cd")} schlepp && ${tok("fn", "python")} ${tok("kw", "-m")} scripts.unpack_chunks /data/schlepp`;
    }

    // "All" selects every box in its group; once they are all selected it reads "None" and clears them.
    function syncToggles() {
      picker.querySelectorAll(".toggle-all").forEach((btn) => {
        const list = btn.closest("fieldset").dataset.list;
        btn.textContent = picked(list).length === total(list) ? "None" : "All";
      });
    }
    picker.addEventListener("change", () => { update(); syncToggles(); });
    picker.querySelectorAll(".toggle-all").forEach((btn) => btn.addEventListener("click", () => {
      const list = btn.closest("fieldset").dataset.list;
      const on = picked(list).length !== total(list);
      boxes(list).forEach((b) => (b.checked = on));
      update();
      syncToggles();
    }));
    update();
    syncToggles();
  }

  // ─── copy buttons on code blocks ─────────────────────────────────────
  const copyIcon = '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2"/></svg>';
  const checkIcon = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>';
  document.querySelectorAll("pre").forEach((pre) => {
    const wrap = document.createElement("div");
    wrap.className = "code-block";
    pre.replaceWith(wrap);
    wrap.append(pre);
    const btn = document.createElement("button");
    btn.className = "copy";
    btn.type = "button";
    btn.setAttribute("aria-label", "Copy code");
    btn.innerHTML = `${copyIcon}<span>Copy</span>`;
    wrap.append(btn);
    let timer;
    btn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(pre.innerText.trim());
        btn.innerHTML = `${checkIcon}<span>Copied</span>`;
      } catch (e) {
        btn.innerHTML = `<span>Press ⌘C</span>`;
      }
      btn.classList.add("done");
      clearTimeout(timer);
      timer = setTimeout(() => {
        btn.classList.remove("done");
        btn.innerHTML = `${copyIcon}<span>Copy</span>`;
      }, 1600);
    });
  });
})();
