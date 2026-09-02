/* Copyright (c) Meta Platforms, Inc. and affiliates. */
// Schlepp project page — minimal interactivity for the sketch.

(function () {
  // ─── hero teaser: one video, six canvas crops ────────────────────────
  const video = document.getElementById("teaser-source");
  const canvases = document.querySelectorAll(".hero-grid canvas.tile-media");

  if (video && canvases.length) {
    initTeaser(video, canvases).catch(() => {
      // fall through silently — the static layout still renders.
    });
  }

  async function initTeaser(video, canvases) {
    const res = await fetch("assets/teaser_layout.json");
    if (!res.ok) throw new Error("teaser layout missing");
    const layout = await res.json();

    // Pair each canvas with its source rect and a 2D context. We size the
    // backing store to the cell's source resolution so the paint is 1:1
    // and CSS handles the on-page scaling.
    const tiles = [];
    canvases.forEach((canvas) => {
      const cell = layout.cells[canvas.dataset.cam];
      if (!cell) return;
      canvas.width = cell.sw;
      canvas.height = cell.sh;
      tiles.push({ canvas, ctx: canvas.getContext("2d"), cell });
    });

    await new Promise((resolve) => {
      if (video.readyState >= 2) resolve();
      else video.addEventListener("loadeddata", resolve, { once: true });
    });

    try {
      await video.play();
    } catch (e) {
      // autoplay blocked — start on first user interaction.
      const start = () => video.play().catch(() => {});
      document.addEventListener("click", start, { once: true });
    }

    let stopped = false;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      video.pause();
      stopped = true;
    }

    function paint() {
      for (const { ctx, cell } of tiles) {
        ctx.drawImage(
          video,
          cell.sx, cell.sy, cell.sw, cell.sh,
          0, 0, cell.sw, cell.sh,
        );
      }
      if (!stopped) requestAnimationFrame(paint);
    }
    paint();
  }

  // ─── viewer: modality + camera pills ─────────────────────────────────
  const viewer = document.querySelector(".viewer");
  if (!viewer) return;

  const label = viewer.querySelector("#viewer-label");
  const caption = viewer.querySelector("#viewer-caption");
  const stage = viewer.querySelector(".viewer-stage");
  const playerVideo = viewer.querySelector("#viewer-video");

  const modalityCopy = {
    rgb: {
      label: "RGB",
      text:
        "8-bit colour, full sensor resolution, 30 fps. Co-registered with " +
        "depth, flow, segmentation, and point tracks at every frame.",
    },
    depth: {
      label: "Depth",
      text:
        "Metric depth in metres, Z-up. Zero marks invalid pixels so your " +
        "loss never silently averages over background.",
    },
    flow_fwd: {
      label: "Forward flow",
      text:
        "Per-frame pixel displacement to the next frame, full resolution, " +
        "with depth-derived validity available out of the box.",
    },
    flow_bwd: {
      label: "Backward flow",
      text:
        "The matching backward field, so cycle-consistency losses and " +
        "occlusion masks are a one-liner.",
    },
    segmentation: {
      label: "Segmentation",
      text:
        "Pass-index labels per pixel, with a separate label dictionary " +
        "that resolves indices to category names per sequence.",
    },
    point_tracks: {
      label: "Point tracks",
      text:
        "~20K 3D-consistent tracks per sequence with per-camera " +
        "visible / in_frustum masks. Stratified across body, cloth, " +
        "carried, and scene categories.",
    },
    obbs: {
      label: "3D OBBs",
      text:
        "Oriented 3D bounding boxes for every object in the scene, with " +
        "category labels and per-frame animation for carried objects.",
    },
  };

  const camCopy = {
    static: "1920 × 1080 · pinhole",
    body_follow: "1920 × 1080 · pinhole",
    object_orbit: "1920 × 1080 · pinhole",
    aria_rgb: "2016 × 1512 · KB4 fisheye",
    aria_slamL: "512 × 512 · KB4 fisheye",
    aria_slamR: "512 × 512 · KB4 fisheye",
  };

  let activeModality = "rgb";
  let activeCam = "body_follow";
  let tourFiles = null;

  // Decide which encoding the browser can play. Probe once, reuse the
  // pick for every subsequent swap so we don't keep asking.
  const probe = document.createElement("video");
  const codec = probe.canPlayType("video/webm; codecs=vp9") ? "webm" : "mp4";

  function render() {
    const m = modalityCopy[activeModality];
    label.textContent = `${activeCam} · ${m.label}`;
    caption.textContent = m.text + " " + `Showing ${activeCam} — ${camCopy[activeCam]}.`;

    if (!tourFiles || !playerVideo) return;
    const entry = tourFiles[activeCam] && tourFiles[activeCam][activeModality];
    if (!entry) return;
    const src = `assets/tour/${entry[codec]}`;
    if (playerVideo.currentSrc.endsWith(entry[codec])) return;

    const resumeAt = playerVideo.currentTime || 0;
    playerVideo.src = src;
    playerVideo.addEventListener("loadedmetadata", () => {
      const dur = playerVideo.duration;
      playerVideo.currentTime = isFinite(dur) ? Math.min(resumeAt, dur - 0.01) : resumeAt;
      playerVideo.play().catch(() => {});
    }, { once: true });
  }

  fetch("assets/tour/_layout.json")
    .then((res) => res.json())
    .then((layout) => {
      tourFiles = layout.files;
      render();
    })
    .catch(() => render());

  viewer.querySelectorAll(".pill").forEach((btn) => {
    btn.addEventListener("click", () => {
      viewer.querySelectorAll(".pill").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      activeModality = btn.dataset.modality;
      render();
    });
  });

  viewer.querySelectorAll(".pill-mini").forEach((btn) => {
    btn.addEventListener("click", () => {
      viewer.querySelectorAll(".pill-mini").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      activeCam = btn.dataset.cam;
      render();
    });
  });

  // ─── viewer: hover-prefetch ──────────────────────────────────────────
  // Warm the HTTP cache for the clip the next click would load, but only
  // when the cursor lingers (≥120ms) and only on real pointers — touch
  // taps fire mouseenter and would prefetch wastefully.
  const prefetched = new Set();
  const finePointer =
    window.matchMedia && window.matchMedia("(hover: hover) and (pointer: fine)").matches;

  function prefetchClip(cam, modality) {
    if (!tourFiles) return;
    const entry = tourFiles[cam] && tourFiles[cam][modality];
    if (!entry) return;
    const url = `assets/tour/${entry[codec]}`;
    if (prefetched.has(url)) return;
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

    viewer.querySelectorAll(".pill").forEach((btn) => {
      btn.addEventListener("mouseenter", () => arm(activeCam, btn.dataset.modality));
      btn.addEventListener("focus", () => arm(activeCam, btn.dataset.modality));
      btn.addEventListener("mouseleave", disarm);
      btn.addEventListener("blur", disarm);
    });
    viewer.querySelectorAll(".pill-mini").forEach((btn) => {
      btn.addEventListener("mouseenter", () => arm(btn.dataset.cam, activeModality));
      btn.addEventListener("focus", () => arm(btn.dataset.cam, activeModality));
      btn.addEventListener("mouseleave", disarm);
      btn.addEventListener("blur", disarm);
    });
  }

  render();

  // ─── BibTeX copy button ──────────────────────────────────────────────
  document.querySelectorAll(".copy-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const target = document.getElementById(btn.dataset.target);
      if (!target) return;
      try {
        await navigator.clipboard.writeText(target.innerText.trim());
        const prev = btn.textContent;
        btn.textContent = "Copied";
        setTimeout(() => (btn.textContent = prev), 1400);
      } catch (e) {
        btn.textContent = "Press ⌘C";
      }
    });
  });
})();
