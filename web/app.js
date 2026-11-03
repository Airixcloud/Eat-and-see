/**
 * ============================================================================
 * Greener - Minimalist Fullscreen 3D Showcase (Awwwards Style)
 * Senior Creative WebGL / Three.js Engineer Implementation
 * ============================================================================
 */

// Global Application State
let currentFile = null;
let webcamStream = null;
let currentDetectedObjects = [];
let isScanning = false;

// 3D Engine Instance
let sphereEngine = null;

// DOM Elements
const cursorGlow = document.getElementById("cursorGlow");
const cursorDot = document.getElementById("cursorDot");
const threeCanvas = document.getElementById("threeCanvas");
const sphereHint = document.getElementById("sphereHint");
const cyberCameraStage = document.getElementById("cyberCameraStage");
const floatingResultHud = document.getElementById("floatingResultHud");
const videoFeed = document.getElementById("videoFeed");
const imageCanvas = document.getElementById("imageCanvas");
const fileInput = document.getElementById("fileInput");
const detectionLayer = document.getElementById("detectionLayer");
const cyberScanLaser = document.getElementById("cyberScanLaser");
const cameraFlashOverlay = document.getElementById("cameraFlashOverlay");
const freezeHintPill = document.getElementById("freezeHintPill");
const cameraStatusChip = document.getElementById("cameraStatusChip");
const scanStatusChip = document.getElementById("scanStatusChip");

let isCameraFrozen = false;

const btnDockCamera = document.getElementById("btnDockCamera");
const btnDockCameraIcon = document.getElementById("btnDockCameraIcon");
const btnDockCameraLabel = document.getElementById("btnDockCameraLabel");
const btnDockReset = document.getElementById("btnDockReset");

const catBadge = document.getElementById("catBadge");
const predItemTitle = document.getElementById("predItemTitle");
const predConfidence = document.getElementById("predConfidence");
const actionGuideBox = document.getElementById("actionGuideBox");
const actionGuideTitle = document.getElementById("actionGuideTitle");
const actionGuideText = document.getElementById("actionGuideText");
const probList = document.getElementById("probList");
const multiObjectSection = document.getElementById("multiObjectSection");
const detectedBadges = document.getElementById("detectedBadges");

const encyclopediaModal = document.getElementById("encyclopediaModal");
const classesGrid = document.getElementById("classesGrid");
const classesCount = document.getElementById("classesCount");
const dropOverlay = document.getElementById("dropOverlay");

// Category Tabs & Detail Lightbox Elements
const classDetailModal = document.getElementById("classDetailModal");
const detailModalIcon = document.getElementById("detailModalIcon");
const detailModalName = document.getElementById("detailModalName");
const detailModalSubItems = document.getElementById("detailModalSubItems");
const detailModalBadge = document.getElementById("detailModalBadge");
const detailGuideText = document.getElementById("detailGuideText");
const sampleImagesGrid = document.getElementById("sampleImagesGrid");

const countAll = document.getElementById("countAll");
const countRecyclable = document.getElementById("countRecyclable");
const countOrganic = document.getElementById("countOrganic");
const countNonRecyclable = document.getElementById("countNonRecyclable");
const countHazardous = document.getElementById("countHazardous");

const catBannerIcon = document.getElementById("catBannerIcon");
const catBannerTitle = document.getElementById("catBannerTitle");
const catBannerText = document.getElementById("catBannerText");

let allLoadedClasses = [];
let currentCategoryFilter = "all";

// ----------------------------------------------------------------------------
// 1. DYNAMIC AMBIENT CURSOR
// ----------------------------------------------------------------------------
let mouseX = window.innerWidth / 2;
let mouseY = window.innerHeight / 2;
let targetGlowX = mouseX;
let targetGlowY = mouseY;

window.addEventListener("mousemove", (e) => {
  mouseX = e.clientX;
  mouseY = e.clientY;
  if (cursorDot) {
    cursorDot.style.left = `${mouseX}px`;
    cursorDot.style.top = `${mouseY}px`;
  }
});

function animateCursor() {
  if (cursorGlow) {
    targetGlowX += (mouseX - targetGlowX) * 0.12;
    targetGlowY += (mouseY - targetGlowY) * 0.12;
    cursorGlow.style.left = `${targetGlowX}px`;
    cursorGlow.style.top = `${targetGlowY}px`;
  }
  requestAnimationFrame(animateCursor);
}
requestAnimationFrame(animateCursor);

// ----------------------------------------------------------------------------
// 2. SENIOR THREE.JS SEGMENTED VOXEL SPHERE (FULLSCREEN SHOWCASE)
// ----------------------------------------------------------------------------
class SegmentedVoxelSphereEngine {
  constructor(canvas) {
    this.canvas = canvas;
    this.width = window.innerWidth;
    this.height = window.innerHeight;

    // States: 'SPHERE' | 'MORPHING' | 'CAMERA_ACTIVE' | 'RESETTING'
    this.state = "SPHERE";

    // Geometry Configuration - Scaled larger for imposing cinematic presence
    this.R = 2.85; // Sphere radius (Hero scale)
    this.RINGS = 16; // Latitude segments (phi)
    this.SEGS = 30; // Longitude segments (theta)
    this.TOTAL_VOXELS = this.RINGS * this.SEGS; // 480 blocks

    // Voxels data array
    this.voxels = [];

    // Animation progress controller (for GSAP)
    this.anim = {
      progress: 0,
      explosion: 0,
    };

    // Color definitions
    this.baseColor = new THREE.Color(0x9aa2aa); // Chrome / metallic silver
    this.neonEmerald = new THREE.Color(0x00ff88); // Neon Emerald Accent
    this.neonCyan = new THREE.Color(0x00d2ff); // Neon Cyan Accent

    // Mouse & Raycasting
    this.mouseNDC = new THREE.Vector2(-999, -999);
    this.isHoveringSphere = false;
    this.raycaster = new THREE.Raycaster();
    this.dummy = new THREE.Object3D();

    this.initScene();
    this.buildInstancedVoxels();
    this.buildAmbientDust();
    this.setupEvents();
    this.animate();
  }

  initScene() {
    this.scene = new THREE.Scene();

    // Perspective Camera
    const aspect = this.width / this.height;
    this.camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 100);
    this.updateCameraDistance();

    // WebGLRenderer with high-end tone mapping & antialiasing
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
      alpha: true,
      powerPreference: "high-performance",
    });
    this.renderer.setSize(this.width, this.height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.2;

    // Master Group for whole sphere (with idle rotation & anti-gravity floating)
    this.sphereGroup = new THREE.Group();
    this.scene.add(this.sphereGroup);

    // Studio Lighting for Chrome Metallic Reflections
    const ambientLight = new THREE.AmbientLight(0x09140e, 1.5);
    this.scene.add(ambientLight);

    const dirKeyLight = new THREE.DirectionalLight(0xffffff, 2.4);
    dirKeyLight.position.set(6, 9, 7);
    this.scene.add(dirKeyLight);

    const dirFillCyan = new THREE.DirectionalLight(0x00d2ff, 1.3);
    dirFillCyan.position.set(-7, -4, 5);
    this.scene.add(dirFillCyan);

    const rimEmerald = new THREE.PointLight(0x00ff88, 4.0, 22);
    rimEmerald.position.set(0, 6, -5);
    this.scene.add(rimEmerald);

    // Interactive pointer specular light
    this.cursorLight = new THREE.PointLight(0x00ff88, 0, 7);
    this.scene.add(this.cursorLight);
  }

  updateCameraDistance() {
    if (this.width < 768) {
      this.camera.position.set(0, 0, 9.4);
    } else if (this.width < 1024) {
      this.camera.position.set(0, 0, 8.2);
    } else {
      this.camera.position.set(0, 0, 7.2);
    }
  }

  buildInstancedVoxels() {
    const boxWidth = 0.38;
    const boxHeight = 0.36;
    const boxDepth = 0.24;
    const boxGeo = new THREE.BoxGeometry(boxWidth, boxHeight, boxDepth);

    const voxelMat = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      metalness: 0.62,
      roughness: 0.22,
      emissive: 0x051610,
      emissiveIntensity: 0.35,
      flatShading: true,
    });

    this.instancedMesh = new THREE.InstancedMesh(boxGeo, voxelMat, this.TOTAL_VOXELS);
    this.instancedMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);

    this.instancedMesh.instanceColor = new THREE.InstancedBufferAttribute(
      new Float32Array(this.TOTAL_VOXELS * 3),
      3
    );

    let idx = 0;
    const R = this.R;

    for (let ring = 0; ring < this.RINGS; ring++) {
      const phi = (Math.PI * (ring + 1)) / (this.RINGS + 1);

      for (let seg = 0; seg < this.SEGS; seg++) {
        const theta = (2 * Math.PI * seg) / this.SEGS;

        const x = R * Math.sin(phi) * Math.cos(theta);
        const y = R * Math.cos(phi);
        const z = R * Math.sin(phi) * Math.sin(theta);

        const basePos = new THREE.Vector3(x, y, z);
        const normal = basePos.clone().normalize();

        const voxel = {
          idx: idx,
          basePos: basePos,
          normal: normal,
          currentPos: basePos.clone(),
          targetRadius: R,
          currentRadius: R,
          baseColor: this.baseColor.clone(),
          currentColor: this.baseColor.clone(),
          targetColor: this.baseColor.clone(),
          randBurst: (Math.random() - 0.5) * 0.5 + 1.2,
        };

        this.voxels.push(voxel);

        this.dummy.position.copy(basePos);
        this.dummy.lookAt(basePos.clone().add(normal));
        this.dummy.updateMatrix();
        this.instancedMesh.setMatrixAt(idx, this.dummy.matrix);
        this.instancedMesh.setColorAt(idx, voxel.baseColor);

        idx++;
      }
    }

    this.instancedMesh.instanceMatrix.needsUpdate = true;
    if (this.instancedMesh.instanceColor) {
      this.instancedMesh.instanceColor.needsUpdate = true;
    }

    this.sphereGroup.add(this.instancedMesh);
  }

  buildAmbientDust() {
    const particleCount = 180;
    const geo = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);

    for (let i = 0; i < particleCount * 3; i += 3) {
      positions[i] = (Math.random() - 0.5) * 16;
      positions[i + 1] = (Math.random() - 0.5) * 12;
      positions[i + 2] = (Math.random() - 0.5) * 12;
    }
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));

    const mat = new THREE.PointsMaterial({
      color: 0x00ff88,
      size: 0.045,
      transparent: true,
      opacity: 0.35,
    });

    this.particles = new THREE.Points(geo, mat);
    this.scene.add(this.particles);
  }

  setupEvents() {
    window.addEventListener("mousemove", (e) => {
      this.mouseNDC.x = (e.clientX / this.width) * 2 - 1;
      this.mouseNDC.y = -(e.clientY / this.height) * 2 + 1;
    });

    window.addEventListener("mouseleave", () => {
      this.mouseNDC.x = -999;
      this.mouseNDC.y = -999;
      this.isHoveringSphere = false;
      this.canvas.style.cursor = "default";
    });

    this.canvas.addEventListener("click", () => {
      if (this.state === "SPHERE") {
        this.morphToCameraStage();
      }
    });

    window.addEventListener("resize", () => this.onResize());
  }

  onResize() {
    this.width = window.innerWidth;
    this.height = window.innerHeight;

    this.camera.aspect = this.width / this.height;
    this.updateCameraDistance();
    this.camera.updateProjectionMatrix();

    this.renderer.setSize(this.width, this.height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  }

  // --------------------------------------------------------------------------
  // MORPHING ANIMATIONS (GSAP)
  // --------------------------------------------------------------------------
  morphToCameraStage() {
    if (this.state !== "SPHERE") return;
    this.state = "MORPHING";

    if (sphereHint) sphereHint.style.opacity = "0";

    gsap.killTweensOf(this.sphereGroup.rotation);
    gsap.to(this.sphereGroup.rotation, {
      x: 0,
      y: 0,
      z: 0,
      duration: 0.5,
      ease: "power2.out",
    });

    const tl = gsap.timeline({
      onComplete: () => {
        this.state = "CAMERA_ACTIVE";
        // Hide 3D voxel cubes completely during camera mode so viewfinder is completely unobstructed
        if (this.instancedMesh) this.instancedMesh.visible = false;

        this.activateWebcamStream();
        updateDockState(true);

        if (cyberCameraStage) {
          cyberCameraStage.style.display = "flex";
          requestAnimationFrame(() => cyberCameraStage.classList.add("active"));
        }
      },
    });

    // Radial blast outward into space
    tl.to(this.anim, {
      explosion: 1.0,
      duration: 0.55,
      ease: "power2.out",
    });
  }

  resetToSphere() {
    if (this.state !== "CAMERA_ACTIVE" && this.state !== "MORPHING") return;
    this.state = "RESETTING";

    this.stopWebcamStream();
    updateDockState(false);

    if (cyberCameraStage) {
      cyberCameraStage.classList.remove("active");
      setTimeout(() => {
        cyberCameraStage.style.display = "none";
      }, 400);
    }

    // Make 3D voxels visible again for magnetic implosion assembly
    if (this.instancedMesh) this.instancedMesh.visible = true;

    const tl = gsap.timeline({
      onComplete: () => {
        this.state = "SPHERE";
        if (sphereHint) sphereHint.style.opacity = "1";
      },
    });

    tl.to(this.anim, {
      explosion: 0.35,
      duration: 0.65,
      ease: "power3.inOut",
    });

    tl.to(this.anim, {
      explosion: 0.0,
      duration: 0.45,
      ease: "elastic.out(1.0, 0.6)",
    });
  }

  // --------------------------------------------------------------------------
  // WEBCAM STREAM INTEGRATION (MIRROR FLIPPED & 100% SHARP)
  // --------------------------------------------------------------------------
  async activateWebcamStream() {
    try {
      webcamStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "environment" },
        audio: false,
      });

      videoFeed.srcObject = webcamStream;
      videoFeed.style.display = "block";
      if (imageCanvas) imageCanvas.style.display = "none";
      await videoFeed.play();

      isCameraFrozen = false;
      if (cyberScanLaser) cyberScanLaser.style.display = "block";
      if (freezeHintPill) freezeHintPill.style.display = "none";
      if (cameraStatusChip) {
        cameraStatusChip.className = "cyber-chip live-chip";
        cameraStatusChip.innerHTML = '<span class="blink-dot"></span> AI VISION ACTIVE';
      }
      if (scanStatusChip) {
        scanStatusChip.innerText = "SMART-SCAN: READY";
      }
    } catch (err) {
      console.warn("Camera could not be accessed directly:", err);
    }
  }

  stopWebcamStream() {
    if (webcamStream) {
      webcamStream.getTracks().forEach((track) => track.stop());
      webcamStream = null;
    }
    isCameraFrozen = false;
    if (freezeHintPill) freezeHintPill.style.display = "none";
    if (detectionLayer) detectionLayer.innerHTML = "";
    if (cyberScanLaser) cyberScanLaser.style.display = "block";
  }

  // --------------------------------------------------------------------------
  // 60 FPS RENDER LOOP
  // --------------------------------------------------------------------------
  animate() {
    requestAnimationFrame(() => this.animate());

    const time = performance.now() * 0.001;

    // 1. Idle rotation & anti-gravity floating
    if (this.state === "SPHERE") {
      this.sphereGroup.rotation.y += 0.005;
      this.sphereGroup.position.y = Math.sin(time * 1.5) * 0.09;
    } else {
      this.sphereGroup.position.y = 0;
    }

    if (this.particles) {
      this.particles.rotation.y = time * 0.015;
    }

    // Proximity Elevation Wave (Raycasting on Sphere)
    let localHit = null;
    if (this.state === "SPHERE" && this.mouseNDC.x !== -999) {
      this.raycaster.setFromCamera(this.mouseNDC, this.camera);

      const invMatrix = this.sphereGroup.matrixWorld.clone().invert();
      const localRay = this.raycaster.ray.clone().applyMatrix4(invMatrix);
      const sphereBound = new THREE.Sphere(new THREE.Vector3(0, 0, 0), this.R);
      const hitPoint = new THREE.Vector3();

      if (localRay.intersectSphere(sphereBound, hitPoint)) {
        localHit = hitPoint;
        this.isHoveringSphere = true;
        this.canvas.style.cursor = "pointer";

        const worldHit = hitPoint.clone().applyMatrix4(this.sphereGroup.matrixWorld);
        this.cursorLight.position.copy(worldHit).add(new THREE.Vector3(0, 0, 0.5));
        this.cursorLight.intensity = 3.0;
      } else {
        this.isHoveringSphere = false;
        this.canvas.style.cursor = "grab";
        this.cursorLight.intensity = 0;
      }
    } else {
      this.cursorLight.intensity = 0;
    }

    // Update 480 Voxel Instances
    const expl = this.anim.explosion;
    const R = this.R;
    const influenceRadius = 1.8;

    for (let i = 0; i < this.TOTAL_VOXELS; i++) {
      const v = this.voxels[i];

      if (this.state === "SPHERE") {
        if (localHit) {
          const dist = localHit.distanceTo(v.basePos);
          if (dist < influenceRadius) {
            const factor = Math.cos((dist / influenceRadius) * Math.PI * 0.5);
            const elevatedRadius = R * (1.0 + 0.35 * Math.pow(factor, 1.4));
            v.targetRadius = elevatedRadius;
            v.targetColor.copy(this.baseColor).lerp(this.neonEmerald, factor * 0.95);
          } else {
            v.targetRadius = R;
            v.targetColor.copy(this.baseColor);
          }
        } else {
          v.targetRadius = R;
          v.targetColor.copy(this.baseColor);
        }

        v.currentRadius += (v.targetRadius - v.currentRadius) * 0.14;
        v.currentColor.lerp(v.targetColor, 0.14);

        v.currentPos.copy(v.normal).multiplyScalar(v.currentRadius);
        this.dummy.position.copy(v.currentPos);
        this.dummy.lookAt(v.currentPos.clone().add(v.normal));
        this.dummy.scale.set(1, 1, 1);
      } else {
        // Disperse voxels outward into space during morph
        const burstRadius = R * (1.0 + expl * (1.8 * v.randBurst));
        const currentPos = v.normal.clone().multiplyScalar(burstRadius);

        this.dummy.position.copy(currentPos);
        this.dummy.lookAt(currentPos.clone().add(v.normal));

        const scaleVal = Math.max(0.001, 1.0 - expl * 0.6);
        this.dummy.scale.setScalar(scaleVal);
      }

      this.dummy.updateMatrix();
      this.instancedMesh.setMatrixAt(i, this.dummy.matrix);
      this.instancedMesh.setColorAt(i, v.currentColor);
    }

    this.instancedMesh.instanceMatrix.needsUpdate = true;
    if (this.instancedMesh.instanceColor) {
      this.instancedMesh.instanceColor.needsUpdate = true;
    }

    this.renderer.render(this.scene, this.camera);
  }
}

// ----------------------------------------------------------------------------
// 3. MINIMAL FLOATING DOCK & HUD UI CONTROLS
// ----------------------------------------------------------------------------
function updateDockState(isCameraActive) {
  if (isCameraActive) {
    btnDockCameraIcon.innerText = "✨";
    btnDockCameraLabel.innerText = "Chụp & Phân Loại AI";
    btnDockReset.style.display = "inline-flex";
  } else {
    btnDockCameraIcon.innerText = "✨";
    btnDockCameraLabel.innerText = "Quét & Phân Loại";
    btnDockReset.style.display = "none";
  }
}

function resetCameraToSphere() {
  if (sphereEngine) {
    sphereEngine.resetToSphere();
  }
}

function toggleResultHud(forceState) {
  if (!floatingResultHud) return;
  if (typeof forceState === "boolean") {
    if (forceState) {
      floatingResultHud.classList.add("active");
    } else {
      floatingResultHud.classList.remove("active");
    }
  } else {
    floatingResultHud.classList.toggle("active");
  }
}

// ----------------------------------------------------------------------------
// 4. ENCYCLOPEDIA MODAL CONTROLS
// ----------------------------------------------------------------------------
function openEncyclopediaModal() {
  if (encyclopediaModal) {
    encyclopediaModal.classList.add("open");
  }
}

function closeEncyclopediaModal() {
  if (encyclopediaModal) {
    encyclopediaModal.classList.remove("open");
  }
}

function handleModalBackdropClick(e) {
  if (e.target === encyclopediaModal) {
    closeEncyclopediaModal();
  }
}

function openClassDetailModal(classId) {
  const item = allLoadedClasses.find((c) => c.id === classId);
  if (!item) return;

  if (detailModalIcon) detailModalIcon.innerText = item.icon || "📦";
  if (detailModalName) detailModalName.innerText = item.name;
  if (detailModalSubItems) {
    detailModalSubItems.innerText = item.sub_items ? `Bao gồm: ${item.sub_items}` : "";
  }
  if (detailModalBadge) {
    detailModalBadge.className = `result-header-badge badge-${item.category}`;
    detailModalBadge.innerText = item.category_title;
  }
  if (detailGuideText) {
    detailGuideText.innerText =
      item.action || "Phân loại vào thùng rác tiêu chuẩn theo chỉ dẫn sinh thái.";
  }

  if (classDetailModal) {
    classDetailModal.style.display = "flex";
    requestAnimationFrame(() => classDetailModal.classList.add("open"));
  }
}

function closeClassDetailModal() {
  if (classDetailModal) {
    classDetailModal.classList.remove("open");
    setTimeout(() => {
      classDetailModal.style.display = "none";
    }, 300);
  }
}

function handleDetailBackdropClick(e) {
  if (e.target === classDetailModal) {
    closeClassDetailModal();
  }
}

async function testWithSampleImageUrl(imgUrl, fileName) {
  closeClassDetailModal();
  closeEncyclopediaModal();

  try {
    const res = await fetch(imgUrl);
    if (!res.ok) throw new Error("Fetch failed");
    const blob = await res.blob();
    const file = new File([blob], fileName || "sample_test.jpg", { type: blob.type || "image/jpeg" });
    loadFile(file);
  } catch (err) {
    console.warn("Direct blob fetch failed, loading via crossOrigin image:", err);
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      if (sphereEngine && sphereEngine.state === "SPHERE") {
        sphereEngine.morphToCameraStage();
      }
      imageCanvas.width = img.width;
      imageCanvas.height = img.height;
      const ctx = imageCanvas.getContext("2d");
      ctx.drawImage(img, 0, 0);

      imageCanvas.style.display = "block";
      videoFeed.style.display = "none";
      if (cyberScanLaser) cyberScanLaser.style.display = "none";
      if (freezeHintPill) freezeHintPill.style.display = "none";
      isCameraFrozen = true;

      if (cyberCameraStage) {
        cyberCameraStage.style.display = "flex";
        requestAnimationFrame(() => cyberCameraStage.classList.add("active"));
      }

      imageCanvas.toBlob((blob) => {
        if (blob) {
          currentFile = new File([blob], fileName || "sample_test.jpg", { type: "image/jpeg" });
          detectAndClassify();
        }
      }, "image/jpeg", 0.95);
    };
    img.src = imgUrl;
  }
}

window.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    closeClassDetailModal();
    closeEncyclopediaModal();
    toggleResultHud(false);
  }
});

// ----------------------------------------------------------------------------
// 5. FULLSCREEN DRAG & DROP & FILE SELECTION
// ----------------------------------------------------------------------------
window.addEventListener("dragover", (e) => {
  e.preventDefault();
  if (dropOverlay) dropOverlay.classList.add("active");
});

window.addEventListener("dragleave", (e) => {
  if (e.clientY <= 0 || e.clientX <= 0 || e.clientX >= window.innerWidth || e.clientY >= window.innerHeight) {
    if (dropOverlay) dropOverlay.classList.remove("active");
  }
});

window.addEventListener("drop", (e) => {
  e.preventDefault();
  if (dropOverlay) dropOverlay.classList.remove("active");
  if (e.dataTransfer.files && e.dataTransfer.files[0]) {
    const file = e.dataTransfer.files[0];
    if (basketballUploadModal && basketballUploadModal.style.display !== "none") {
      shootBasketballFile(file);
    } else {
      openBasketballModal();
      setTimeout(() => shootBasketballFile(file), 300);
    }
  }
});

function triggerFileInput() {
  openBasketballModal();
}

function triggerNativeFileInput() {
  if (fileInput) {
    fileInput.value = ""; // Crucial: allow re-selecting same file or switching files every time!
    fileInput.click();
  }
}

function handleFileSelect(e) {
  if (e.target.files && e.target.files[0]) {
    const file = e.target.files[0];
    onRealFileSelected(file);
  }
}

function loadFile(file) {
  currentFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    const img = new Image();
    img.onload = () => {
      if (sphereEngine && sphereEngine.state === "SPHERE") {
        sphereEngine.morphToCameraStage();
      }

      imageCanvas.width = img.width;
      imageCanvas.height = img.height;
      const ctx = imageCanvas.getContext("2d");
      ctx.drawImage(img, 0, 0);

      imageCanvas.style.display = "block";
      videoFeed.style.display = "none";

      // Static image mode: turn off laser beam & freeze
      if (cyberScanLaser) cyberScanLaser.style.display = "none";
      if (freezeHintPill) freezeHintPill.style.display = "none";
      isCameraFrozen = true;
      if (cameraStatusChip) {
        cameraStatusChip.className = "cyber-chip";
        cameraStatusChip.innerHTML = "ẢNH TẢI LÊN";
      }
      if (scanStatusChip) {
        scanStatusChip.innerText = "CHẾ ĐỘ TẢI ẢNH";
      }

      if (cyberCameraStage) {
        cyberCameraStage.style.display = "flex";
        requestAnimationFrame(() => cyberCameraStage.classList.add("active"));
      }

      detectAndClassify();
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

// ----------------------------------------------------------------------------
// 6. SAMPLE IMAGE LOADER
// ----------------------------------------------------------------------------
async function loadSampleImage() {
  try {
    let res = await fetch("/static/test_bottle.jpg");
    if (!res.ok) {
      res = await fetch("/test_bottle.jpg");
    }
    const blob = await res.blob();
    const file = new File([blob], "test_bottle.jpg", { type: "image/jpeg" });
    loadFile(file);
  } catch (err) {
    console.error("Error loading sample image:", err);
  }
}

// ----------------------------------------------------------------------------
// 7. PRIMARY ACTION HANDLER (CAMERA TRIGGER, FREEZE & AI CLASSIFICATION)
// ----------------------------------------------------------------------------
async function handlePrimaryAction() {
  if (!sphereEngine) return;

  if (sphereEngine.state === "SPHERE") {
    sphereEngine.morphToCameraStage();
    return;
  }

  if (sphereEngine.state === "CAMERA_ACTIVE") {
    if (isCameraFrozen) {
      // Đang ở màn hình dừng -> Mở lại camera trực tiếp để quét tiếp
      resumeCameraStream();
    } else {
      // Đang xem camera -> Bấm vào quét: tắt dòng laser lên xuống & dừng màn hình lúc đó luôn
      freezeCameraAndCapture();
    }
  }
}

function freezeCameraAndCapture() {
  if (!videoFeed || videoFeed.videoWidth === 0) return;

  // 1. Chớp sáng màn trập phong cách Sci-Fi
  if (cameraFlashOverlay) {
    cameraFlashOverlay.classList.add("flash");
    setTimeout(() => cameraFlashOverlay.classList.remove("flash"), 160);
  }

  // 2. TẮT DÒNG CHẠY LÊN XUỐNG NGAY LẬP TỨC
  if (cyberScanLaser) {
    cyberScanLaser.style.display = "none";
  }

  // 3. DỪNG MÀN HÌNH LÚC ĐÓ LUÔN (FREEZE FRAME)
  if (!videoFeed.paused) {
    videoFeed.pause();
  }
  isCameraFrozen = true;

  // Cập nhật nhãn trạng thái HUD
  if (cameraStatusChip) {
    cameraStatusChip.className = "cyber-chip";
    cameraStatusChip.innerHTML = '<span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:#EF4444;box-shadow:0 0 8px #EF4444;margin-right:6px;"></span> ĐÃ DỪNG HÌNH';
  }
  if (scanStatusChip) {
    scanStatusChip.innerText = "AI ĐANG PHÂN TÍCH...";
  }
  if (freezeHintPill) {
    freezeHintPill.style.display = "flex";
  }

  // Chụp đúng khung hình vừa dừng đó để gửi lên mạng nơ-ron AI
  imageCanvas.width = videoFeed.videoWidth;
  imageCanvas.height = videoFeed.videoHeight;
  const ctx = imageCanvas.getContext("2d");

  // Lật gương ngang để khớp tuyệt đối với góc nhìn scaleX(-1) của người dùng
  ctx.save();
  ctx.translate(imageCanvas.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(videoFeed, 0, 0, imageCanvas.width, imageCanvas.height);
  ctx.restore();

  imageCanvas.toBlob((blob) => {
    currentFile = new File([blob], "camera_capture.jpg", { type: "image/jpeg" });
    detectAndClassify();
  }, "image/jpeg", 0.95);
}

function resumeCameraStream() {
  if (!videoFeed) return;

  // 1. Bỏ dừng hình và tiếp tục phát live webcam
  videoFeed.style.display = "block";
  if (imageCanvas) imageCanvas.style.display = "none";
  videoFeed.play().catch(e => console.log("Video resume play error:", e));
  isCameraFrozen = false;

  // 2. BẬT LẠI DÒNG QUÉT LÊN XUỐNG
  if (cyberScanLaser) {
    cyberScanLaser.style.display = "block";
  }

  // 3. Xóa các bounding box cũ và ẩn nút hint
  if (detectionLayer) {
    detectionLayer.innerHTML = "";
  }
  if (freezeHintPill) {
    freezeHintPill.style.display = "none";
  }

  // 4. Khôi phục nhãn HUD & Nút Dock
  if (cameraStatusChip) {
    cameraStatusChip.className = "cyber-chip live-chip";
    cameraStatusChip.innerHTML = '<span class="blink-dot"></span> AI VISION ACTIVE';
  }
  if (scanStatusChip) {
    scanStatusChip.innerText = "SMART-SCAN: READY";
  }
  btnDockCameraIcon.innerText = "✨";
  btnDockCameraLabel.innerText = "Chụp & Phân Loại AI";
}

// ----------------------------------------------------------------------------
// 8. AI INFERENCE & BACKEND API CALL
// ----------------------------------------------------------------------------
async function detectAndClassify() {
  if (!currentFile || isScanning) return;

  isScanning = true;
  toggleResultHud(true); // Automatically slide open Floating HUD for diagnosis

  btnDockCameraLabel.innerText = "Đang nhận diện AI...";
  predItemTitle.innerText = "Đang phân tích...";
  predConfidence.innerText = "Vui lòng chờ giây lát...";

  const formData = new FormData();
  formData.append("file", currentFile);

  try {
    const res = await fetch("/api/detect_and_classify", {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      throw new Error(`Server returned status ${res.status}`);
    }

    const data = await res.json();
    renderResults(data);
  } catch (err) {
    console.error("Classification error:", err);
    predItemTitle.innerText = "Lỗi xử lý";
    predConfidence.innerText = err.message;
  } finally {
    isScanning = false;
    if (sphereEngine && sphereEngine.state === "CAMERA_ACTIVE") {
      if (isCameraFrozen) {
        btnDockCameraIcon.innerText = "🔄";
        btnDockCameraLabel.innerText = "Quét Lại / Tiếp Tục";
        if (scanStatusChip) scanStatusChip.innerText = "QUÉT HOÀN TẤT";
      } else {
        btnDockCameraIcon.innerText = "✨";
        btnDockCameraLabel.innerText = "Chụp & Phân Loại AI";
      }
    } else {
      btnDockCameraIcon.innerText = "✨";
      btnDockCameraLabel.innerText = "Quét & Phân Loại";
    }
  }
}

// ----------------------------------------------------------------------------
// 9. RENDER RESULTS IN FLOATING GLASS HUD & BOUNDING BOXES
// ----------------------------------------------------------------------------
function renderResults(data) {
  if (detectionLayer) detectionLayer.innerHTML = "";
  const detected = data.detected_objects || [];
  currentDetectedObjects = detected;

  let primary = detected.length > 0 ? detected[0] : data.whole_image;
  applyActivePrediction(primary);

  if (detected.length > 0) {
    multiObjectSection.style.display = "block";
    detectedBadges.innerHTML = "";

    detected.forEach((obj) => {
      const [nx1, ny1, nx2, ny2] = obj.normalized_box;

      if (detectionLayer) {
        const boxElem = document.createElement("div");
        boxElem.className = `bbox bbox-${obj.category}`;
        boxElem.style.left = `${nx1 * 100}%`;
        boxElem.style.top = `${ny1 * 100}%`;
        boxElem.style.width = `${(nx2 - nx1) * 100}%`;
        boxElem.style.height = `${(ny2 - ny1) * 100}%`;

        const tag = document.createElement("div");
        tag.className = "bbox-tag";
        tag.style.backgroundColor = obj.category_color;
        tag.innerText = `${obj.display_name} (${obj.confidence}%)`;
        boxElem.appendChild(tag);

        boxElem.addEventListener("click", () => applyActivePrediction(obj));
        detectionLayer.appendChild(boxElem);
      }

      const badgeBtn = document.createElement("button");
      badgeBtn.className = "tech-badge";
      badgeBtn.style.cursor = "pointer";
      badgeBtn.innerHTML = `<span>🎯</span> ${obj.display_name} - ${obj.confidence}%`;
      badgeBtn.onclick = () => applyActivePrediction(obj);
      detectedBadges.appendChild(badgeBtn);
    });
  } else {
    multiObjectSection.style.display = "none";
  }
}

function applyActivePrediction(pred) {
  if (!pred) return;

  predItemTitle.innerHTML = `${pred.display_name}`;
  predConfidence.innerHTML = `Độ tin cậy: <strong style="color: ${pred.category_color}">${pred.confidence}%</strong> ${
    pred.is_low_confidence ? ' <span style="font-size:0.75rem; color:#f59e0b;">(Cần xác nhận)</span>' : ""
  }`;

  catBadge.className = `result-header-badge badge-${pred.category}`;
  catBadge.innerText = pred.category_title;

  actionGuideBox.className = `action-guide-box guide-${pred.category}`;
  actionGuideTitle.innerText = `HƯỚNG DẪN XỬ LÝ - ${pred.category_title.toUpperCase()}`;
  actionGuideText.innerText = pred.category_action;

  const topList = pred.top_predictions || [];
  if (topList.length > 0) {
    probList.innerHTML = topList
      .map(
        (item) => `
      <div class="prob-row">
        <div class="prob-labels">
          <span class="prob-name">${item.display_name}</span>
          <span class="prob-val">${item.confidence}%</span>
        </div>
        <div class="prob-track">
          <div class="prob-fill" style="width: ${item.confidence}%; background: linear-gradient(90deg, ${
          item.category_info ? item.category_info.color : "#0284c7"
        }, #38bdf8);"></div>
        </div>
      </div>
    `
      )
      .join("");
  } else {
    probList.innerHTML = `<div class="prob-empty">Phát hiện thiết bị phần cứng từ YOLOv8</div>`;
  }
}

// ----------------------------------------------------------------------------
// 10. LOAD 30+ CLASSES ENCYCLOPEDIA
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// 10. LOAD 30+ CLASSES ENCYCLOPEDIA & CATEGORY FILTERING
// ----------------------------------------------------------------------------
function filterClasses(category) {
  currentCategoryFilter = category;

  // Update tabs active state
  const tabs = document.querySelectorAll("#categoryTabsBar .cat-tab");
  tabs.forEach((tab) => {
    if (tab.getAttribute("data-cat") === category) {
      tab.classList.add("active");
    } else {
      tab.classList.remove("active");
    }
  });

  // Update Category Guidance Banner
  if (catBannerTitle && catBannerText && catBannerIcon) {
    switch (category) {
      case "recyclable":
        catBannerIcon.innerText = "♻️";
        catBannerTitle.innerText = "HƯỚNG DẪN - RÁC TÁI CHẾ";
        catBannerText.innerText =
          "Bao gồm chai nhựa, lon nhôm, thùng carton, giấy báo, đồ hộp thiếc... Cần làm sạch, phơi khô và ép phẳng trước khi bỏ vào thùng rác màu xanh lá.";
        break;
      case "organic":
        catBannerIcon.innerText = "🍂";
        catBannerTitle.innerText = "HƯỚNG DẪN - RÁC HỮU CƠ";
        catBannerText.innerText =
          "Thức ăn thừa, bã cà phê, vỏ trứng, túi lọc trà... Dùng để ủ phân compost vi sinh hoặc bỏ vào thùng rác hữu cơ màu vàng/cam.";
        break;
      case "non-recyclable":
        catBannerIcon.innerText = "🚫";
        catBannerTitle.innerText = "HƯỚNG DẪN - RÁC VÔ CƠ / KHÔNG THỂ TÁI CHẾ";
        catBannerText.innerText =
          "Hộp xốp, ly giấy tráng PE, túi rác, ống hút nhựa, dao nĩa dùng 1 lần... Bỏ vào thùng rác vô cơ sinh hoạt để chôn lấp/xử lý nhiệt.";
        break;
      case "hazardous":
        catBannerIcon.innerText = "⚡";
        catBannerTitle.innerText = "HƯỚNG DẪN - RÁC ĐIỆN TỬ / NGUY HẠI";
        catBannerText.innerText =
          "Thiết bị số, linh kiện máy tính, điện thoại, vi mạch... Tuyệt đối KHÔNG vứt chung rác sinh hoạt. Nộp tại điểm thu gom rác điện tử.";
        break;
      default:
        catBannerIcon.innerText = "🌐";
        catBannerTitle.innerText = "DANH MỤC PHÂN LOẠI RÁC SINH THÁI";
        catBannerText.innerText =
          "Bấm vào từng loại rác để xem chi tiết cách phân loại chuẩn sinh thái.";
        break;
    }
  }

  renderClassesGrid();
}

function renderClassesGrid() {
  if (!classesGrid) return;

  const filtered =
    currentCategoryFilter === "all"
      ? allLoadedClasses
      : allLoadedClasses.filter((c) => c.category === currentCategoryFilter);

  classesGrid.innerHTML = filtered
    .map(
      (item) => `
    <div class="class-pill" onclick="openClassDetailModal('${item.id}')" title="Bấm để xem hướng dẫn của ${item.name}">
      <div class="class-pill-header">
        <span class="class-pill-icon">${item.icon || "📦"}</span>
        <span class="class-pill-badge badge-${item.category}">${item.category_title}</span>
      </div>
      <div class="class-pill-name">${item.name}</div>
      ${item.sub_items ? `<div class="class-pill-sub">${item.sub_items}</div>` : ""}
      <div class="class-pill-action-hint">
        <span>📖 Xem hướng dẫn xử lý</span>
        <span class="pill-arrow">→</span>
      </div>
    </div>
  `
    )
    .join("");
}

async function loadEncyclopedia() {
  try {
    const res = await fetch("/api/classes");
    const items = await res.json();
    allLoadedClasses = items;

    if (classesCount) {
      classesCount.innerText = `${items.length} Nhóm Rác Chuẩn`;
    }

    // Update count numbers in category tabs
    const cAll = items.length;
    const cRecyclable = items.filter((c) => c.category === "recyclable").length;
    const cOrganic = items.filter((c) => c.category === "organic").length;
    const cNonRecyclable = items.filter((c) => c.category === "non-recyclable").length;
    const cHazardous = items.filter((c) => c.category === "hazardous").length;

    if (countAll) countAll.innerText = cAll;
    if (countRecyclable) countRecyclable.innerText = cRecyclable;
    if (countOrganic) countOrganic.innerText = cOrganic;
    if (countNonRecyclable) countNonRecyclable.innerText = cNonRecyclable;
    if (countHazardous) countHazardous.innerText = cHazardous;

    renderClassesGrid();
  } catch (err) {
    console.error("Failed to load encyclopedia classes:", err);
  }
}

// ----------------------------------------------------------------------------
// 11. BASKETBALL UPLOAD ANIMATION (TIKTOK @paulcodes4 RECREATION)
// ----------------------------------------------------------------------------
let isBballSoundEnabled = true;
let bballUploadCount = 0;
let isBballShooting = false;
let isBballAiming = false;
let bballAimStart = { x: 0, y: 0 };
let currentTargetFile = null;

const basketballUploadModal = document.getElementById("basketballUploadModal");
const bballDropzone = document.getElementById("bballDropzone");
const bballSuccessCard = document.getElementById("bballSuccessCard");
const bballProgressBar = document.getElementById("bballProgressBar");
const bballSuccessName = document.getElementById("bballSuccessName");
const bballSuccessSize = document.getElementById("bballSuccessSize");
const bballSuccessBadge = document.getElementById("bballSuccessBadge");
const bballDropPrompt = document.getElementById("bballDropPrompt");
const bballNetMesh = document.getElementById("bballNetMesh");
const bballTrajectorySvg = document.getElementById("bballTrajectorySvg");
const bballFlyingItem = document.getElementById("bballFlyingItem");
const bballFlyingLabel = document.getElementById("bballFlyingLabel");
const bballDraggableItem = document.getElementById("bballDraggableItem");
const bballInstruction = document.getElementById("bballInstruction");
const bballCount = document.getElementById("bballCount");
const bballParticles = document.getElementById("bballParticles");

function toggleUploadSound() {
  isBballSoundEnabled = !isBballSoundEnabled;
  const icon = document.getElementById("bballSoundIcon");
  if (icon) {
    icon.innerText = isBballSoundEnabled ? "🔊" : "🔇";
  }
}

function playBballSwishSound() {
  if (!isBballSoundEnabled) return;
  try {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (!AudioContext) return;
    const ctx = new AudioContext();

    // 1. Friction net swish (bandpassed noise)
    const bufferSize = ctx.sampleRate * 0.22;
    const buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
    const output = buffer.getChannelData(0);
    for (let i = 0; i < bufferSize; i++) {
      output[i] = (Math.random() * 2 - 1) * Math.exp(-i / (bufferSize * 0.25));
    }

    const whiteNoise = ctx.createBufferSource();
    whiteNoise.buffer = buffer;

    const filter = ctx.createBiquadFilter();
    filter.type = "bandpass";
    filter.frequency.setValueAtTime(2400, ctx.currentTime);
    filter.frequency.exponentialRampToValueAtTime(750, ctx.currentTime + 0.2);
    filter.Q.value = 3.5;

    const gainNode = ctx.createGain();
    gainNode.gain.setValueAtTime(0.55, ctx.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.2);

    whiteNoise.connect(filter);
    filter.connect(gainNode);
    gainNode.connect(ctx.destination);
    whiteNoise.start();

    // 2. Subtle soft basket thud
    const osc = ctx.createOscillator();
    const oscGain = ctx.createGain();
    osc.type = "sine";
    osc.frequency.setValueAtTime(150, ctx.currentTime + 0.04);
    osc.frequency.exponentialRampToValueAtTime(45, ctx.currentTime + 0.16);
    oscGain.gain.setValueAtTime(0.25, ctx.currentTime + 0.04);
    oscGain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.16);

    osc.connect(oscGain);
    oscGain.connect(ctx.destination);
    osc.start(ctx.currentTime + 0.04);
    osc.stop(ctx.currentTime + 0.18);
  } catch (err) {
    console.warn("Audio synthesis error:", err);
  }
}

function openBasketballModal() {
  if (!basketballUploadModal) return;
  basketballUploadModal.style.display = "flex";
  requestAnimationFrame(() => basketballUploadModal.classList.add("open"));

  // Reset internal states
  if (bballSuccessCard) bballSuccessCard.style.display = "none";
  if (bballDropPrompt) bballDropPrompt.style.opacity = "1";
  if (bballDraggableItem) {
    bballDraggableItem.style.display = "block";
    bballDraggableItem.style.transform = "";
  }
  if (bballFlyingItem) bballFlyingItem.style.display = "none";

  const addBtnLabel = document.getElementById("bballAddFileLabel");
  const addBtnIcon = document.getElementById("bballAddFileIcon");

  if (currentTargetFile) {
    if (addBtnLabel) addBtnLabel.innerText = "Đổi File";
    if (addBtnIcon) addBtnIcon.innerText = "🔄";
    if (bballInstruction) {
      bballInstruction.innerText = "Kéo bóng lên & thả tay để ném rổ!";
      bballInstruction.classList.remove("shooting-active");
    }
  } else {
    if (addBtnLabel) addBtnLabel.innerText = "Thêm File";
    if (addBtnIcon) addBtnIcon.innerText = "➕";
    if (bballInstruction) {
      bballInstruction.innerText = "Kéo thả để ngắm & ném bóng vào rổ";
      bballInstruction.classList.remove("shooting-active");
    }
  }
  clearTrajectory();
  isBballShooting = false;
  isBballAiming = false;
}

function onRealFileSelected(file) {
  if (!file) return;
  currentTargetFile = file;

  // Update top-right button to "Đổi File"
  const addBtnLabel = document.getElementById("bballAddFileLabel");
  const addBtnIcon = document.getElementById("bballAddFileIcon");
  if (addBtnLabel) addBtnLabel.innerText = "Đổi File";
  if (addBtnIcon) addBtnIcon.innerText = "🔄";

  // Mount real thumbnail onto draggable launchpad card and flying item
  const thumbPreview = document.getElementById("bballThumbPreview");
  const flyingThumb = document.getElementById("bballFlyingThumb");
  const flyingIcon = document.getElementById("bballFlyingIcon");
  const ballIcon = document.getElementById("bballBallIcon");

  if (file.type && file.type.startsWith("image/")) {
    const reader = new FileReader();
    reader.onload = (ev) => {
      const dataUrl = ev.target.result;
      if (thumbPreview) {
        thumbPreview.src = dataUrl;
        thumbPreview.style.display = "block";
      }
      if (flyingThumb) {
        flyingThumb.src = dataUrl;
        flyingThumb.style.display = "block";
      }
      if (ballIcon) ballIcon.style.display = "none";
      if (flyingIcon) flyingIcon.style.display = "none";

      // If upload modal is open, trigger the fast snappy shot with the real file!
      if (basketballUploadModal && basketballUploadModal.style.display !== "none") {
        setTimeout(() => {
          shootBasketballFile(file);
        }, 60);
      } else {
        loadFile(file);
      }
    };
    reader.readAsDataURL(file);
  } else {
    if (basketballUploadModal && basketballUploadModal.style.display !== "none") {
      shootBasketballFile(file);
    } else {
      loadFile(file);
    }
  }
}

function closeBasketballModal() {
  if (!basketballUploadModal) return;
  basketballUploadModal.classList.remove("open");
  setTimeout(() => {
    basketballUploadModal.style.display = "none";
  }, 350);
}

function handleUploadBackdropClick(e) {
  if (e.target === basketballUploadModal) {
    closeBasketballModal();
  }
}

function clearTrajectory() {
  if (bballTrajectorySvg) bballTrajectorySvg.innerHTML = "";
}

function drawTrajectory(startX, startY, hoopX, hoopY, pullFactor) {
  if (!bballTrajectorySvg) return;

  const ctrlX = (startX + hoopX) / 2 + (pullFactor.x || 0) * 0.35;
  const ctrlY = Math.min(hoopY - 55, (startY + hoopY) / 2 - 75);

  let dotsHtml = "";
  const numDots = 10;
  for (let i = 1; i <= numDots; i++) {
    const t = i / (numDots + 1);
    const x = (1 - t) * (1 - t) * startX + 2 * (1 - t) * t * ctrlX + t * t * hoopX;
    const y = (1 - t) * (1 - t) * startY + 2 * (1 - t) * t * ctrlY + t * t * hoopY;
    const r = 2.5 + (1 - t) * 1.5;
    const opacity = 0.45 + (1 - t) * 0.55;
    dotsHtml += `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${r.toFixed(1)}" fill="#00FF88" opacity="${opacity.toFixed(2)}" filter="drop-shadow(0 0 4px #00FF88)" />`;
  }
  bballTrajectorySvg.innerHTML = dotsHtml;
}

function shootBasketballFile(file) {
  if (isBballShooting) return;
  isBballShooting = true;
  currentTargetFile = file;

  clearTrajectory();
  if (bballInstruction) {
    bballInstruction.innerText = "SWISH! Đang ném vào rổ...";
    bballInstruction.classList.remove("shooting-active");
  }

  // File label info from real file
  const fileName = file ? file.name : "anh_cua_ban.jpg";
  const ext = fileName.split(".").pop().toUpperCase().slice(0, 4) || "ẢNH";
  const sizeMb = file ? (file.size / (1024 * 1024)).toFixed(1) + " MB" : "1.5 MB";

  if (bballFlyingLabel) bballFlyingLabel.innerText = ext;
  if (bballSuccessBadge) bballSuccessBadge.innerText = ext;
  if (bballSuccessName) bballSuccessName.innerText = fileName;
  if (bballSuccessSize) bballSuccessSize.innerText = sizeMb;

  // Real Image preview handling
  const thumbPreview = document.getElementById("bballThumbPreview");
  const flyingThumb = document.getElementById("bballFlyingThumb");
  const flyingIcon = document.getElementById("bballFlyingIcon");

  if (file && file.type && file.type.startsWith("image/")) {
    const reader = new FileReader();
    reader.onload = (e) => {
      const dataUrl = e.target.result;
      if (flyingThumb) {
        flyingThumb.src = dataUrl;
        flyingThumb.style.display = "block";
      }
      if (flyingIcon) flyingIcon.style.display = "none";
      if (thumbPreview) {
        thumbPreview.src = dataUrl;
        thumbPreview.style.display = "block";
      }
    };
    reader.readAsDataURL(file);
  }

  // Calculate coordinates relative to dropzone
  const dzRect = bballDropzone.getBoundingClientRect();
  const hoopEl = document.getElementById("bballRimRing");
  const hoopRect = hoopEl.getBoundingClientRect();
  const targetX = hoopRect.left + hoopRect.width / 2 - dzRect.left;
  const targetY = hoopRect.top + hoopRect.height / 2 - dzRect.top;

  const dragRect = bballDraggableItem.getBoundingClientRect();
  const startX = dragRect.left + dragRect.width / 2 - dzRect.left;
  const startY = dragRect.top + dragRect.height / 2 - dzRect.top;

  bballDraggableItem.style.display = "none";
  bballFlyingItem.style.display = "block";
  bballFlyingItem.style.left = `${startX - 26}px`;
  bballFlyingItem.style.top = `${startY - 32}px`;

  const dx = targetX - startX;
  const dy = targetY - startY;

  // Ultra-Fast & Silky Smooth Coordinated GSAP Flight Arc (~0.22s)
  const tl = gsap.timeline({
    onComplete: () => {
      onBasketScored(file);
    }
  });

  // Smooth concurrent X, Y, Scale & Rotation for natural physical motion
  tl.to(bballFlyingItem, {
    x: dx,
    duration: 0.22,
    ease: "power1.inOut"
  }, 0);

  tl.to(bballFlyingItem, {
    scale: 0.72,
    duration: 0.22,
    ease: "power1.out"
  }, 0);

  tl.to(bballFlyingItem, {
    rotation: -10,
    duration: 0.13,
    ease: "sine.out"
  }, 0);

  // Y trajectory: Rise smoothly to apex above hoop (0.13s)
  tl.to(bballFlyingItem, {
    y: dy - 48,
    duration: 0.13,
    ease: "power2.out"
  }, 0);

  // Y trajectory: Natural gravity plunge straight through net (0.09s)
  tl.to(bballFlyingItem, {
    y: dy + 30,
    rotation: 0,
    duration: 0.09,
    ease: "power2.in",
    onStart: () => {
      playBballSwishSound();
      if (bballNetMesh) {
        bballNetMesh.classList.remove("swish");
        void bballNetMesh.offsetWidth; // trigger reflow
        bballNetMesh.classList.add("swish");
      }
      triggerBballConfetti(targetX, targetY);
    }
  }, 0.13);
}

function onBasketScored(file) {
  if (bballFlyingItem) bballFlyingItem.style.display = "none";

  // Hide drop prompt, show success card
  if (bballDropPrompt) bballDropPrompt.style.opacity = "0";
  if (bballSuccessCard) {
    bballSuccessCard.style.display = "flex";
    if (bballProgressBar) {
      bballProgressBar.style.width = "0%";
      requestAnimationFrame(() => {
        bballProgressBar.style.width = "100%";
      });
    }
  }

  // Snappy delay (380ms): close modal and immediately run AI Vision on the REAL uploaded file!
  setTimeout(() => {
    closeBasketballModal();
    if (file) {
      loadFile(file);
    } else {
      triggerNativeFileInput();
    }
  }, 380);
}

function triggerBballConfetti(originX, originY) {
  if (!bballParticles) return;
  const colors = ["#00FF88", "#00E5FF", "#55FF99", "#FFFFFF", "#10B981"];
  for (let i = 0; i < 16; i++) {
    const p = document.createElement("div");
    p.className = "bball-particle";
    p.style.backgroundColor = colors[i % colors.length];
    p.style.boxShadow = `0 0 6px ${colors[i % colors.length]}`;
    p.style.left = `${originX}px`;
    p.style.top = `${originY}px`;
    bballParticles.appendChild(p);

    const angle = (Math.PI * 2 * i) / 16 + (Math.random() - 0.5) * 0.4;
    const dist = 35 + Math.random() * 45;
    const destX = Math.cos(angle) * dist;
    const destY = Math.sin(angle) * dist - 15;

    gsap.to(p, {
      x: destX,
      y: destY,
      opacity: 0,
      scale: 0.3,
      duration: 0.35 + Math.random() * 0.15,
      ease: "power2.out",
      onComplete: () => p.remove(),
    });
  }
}

function setupBasketballDragEvents() {
  if (!bballDraggableItem || !bballDropzone) return;

  let pointerHasMoved = false;

  const onPointerDown = (e) => {
    if (isBballShooting) return;
    pointerHasMoved = false;
    isBballAiming = true;
    const clientX = e.clientX !== undefined ? e.clientX : (e.touches && e.touches[0].clientX);
    const clientY = e.clientY !== undefined ? e.clientY : (e.touches && e.touches[0].clientY);
    bballAimStart = { x: clientX, y: clientY };
    bballDraggableItem.classList.add("dragging");

    if (bballInstruction) {
      bballInstruction.innerText = "Release to shoot!";
      bballInstruction.classList.add("shooting-active");
    }
  };

  const onPointerMove = (e) => {
    if (!isBballAiming || isBballShooting) return;
    const clientX = e.clientX !== undefined ? e.clientX : (e.touches && e.touches[0].clientX);
    const clientY = e.clientY !== undefined ? e.clientY : (e.touches && e.touches[0].clientY);

    const pullX = clientX - bballAimStart.x;
    const pullY = clientY - bballAimStart.y;
    if (Math.abs(pullX) > 4 || Math.abs(pullY) > 4) {
      pointerHasMoved = true;
    }

    const dzRect = bballDropzone.getBoundingClientRect();
    const hoopEl = document.getElementById("bballRimRing");
    const hoopRect = hoopEl.getBoundingClientRect();
    const targetX = hoopRect.left + hoopRect.width / 2 - dzRect.left;
    const targetY = hoopRect.top + hoopRect.height / 2 - dzRect.top;

    const dragRect = bballDraggableItem.getBoundingClientRect();
    const startX = dragRect.left + dragRect.width / 2 - dzRect.left;
    const startY = dragRect.top + dragRect.height / 2 - dzRect.top;

    bballDraggableItem.style.transform = `translate(${pullX * 0.35}px, ${pullY * 0.35}px) scale(1.08)`;
    drawTrajectory(startX, startY, targetX, targetY, { x: pullX, y: pullY });
  };

  const onPointerUp = (e) => {
    if (!isBballAiming || isBballShooting) return;
    isBballAiming = false;
    bballDraggableItem.classList.remove("dragging");

    // If simply clicked without dragging -> trigger file selection immediately!
    if (!pointerHasMoved && !currentTargetFile) {
      clearTrajectory();
      bballDraggableItem.style.transform = "";
      triggerNativeFileInput();
      return;
    }

    const clientY = e.clientY !== undefined ? e.clientY : (e.changedTouches && e.changedTouches[0].clientY);
    const deltaY = bballAimStart.y - clientY;

    if (deltaY > 15 || bballAimStart.y - clientY > -40) {
      if (currentTargetFile) {
        shootBasketballFile(currentTargetFile);
      } else {
        // No file yet -> open real file selector!
        triggerNativeFileInput();
      }
    } else {
      gsap.to(bballDraggableItem, {
        x: 0,
        y: 0,
        duration: 0.2,
        ease: "elastic.out(1, 0.5)",
        onComplete: () => {
          bballDraggableItem.style.transform = "";
        },
      });
      clearTrajectory();
      if (bballInstruction) {
        bballInstruction.innerText = "Chọn ảnh của bạn & kéo thả để ném vào rổ AI";
        bballInstruction.classList.remove("shooting-active");
      }
    }
  };

  // Mouse events
  bballDraggableItem.addEventListener("mousedown", onPointerDown);
  window.addEventListener("mousemove", onPointerMove);
  window.addEventListener("mouseup", onPointerUp);

  // Touch events
  bballDraggableItem.addEventListener("touchstart", onPointerDown, { passive: true });
  window.addEventListener("touchmove", onPointerMove, { passive: true });
  window.addEventListener("touchend", onPointerUp);

  // Direct dropzone drop: accepts REAL user image files directly!
  bballDropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    bballDropzone.classList.add("drag-active");
  });
  bballDropzone.addEventListener("dragleave", () => {
    bballDropzone.classList.remove("drag-active");
  });
  bballDropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    bballDropzone.classList.remove("drag-active");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const realFile = e.dataTransfer.files[0];
      shootBasketballFile(realFile);
    }
  });
}

// ----------------------------------------------------------------------------
// 12. INITIALIZATION
// ----------------------------------------------------------------------------
window.addEventListener("DOMContentLoaded", () => {
  if (threeCanvas && window.THREE && window.gsap) {
    sphereEngine = new SegmentedVoxelSphereEngine(threeCanvas);
  } else {
    console.error("Three.js or GSAP not loaded properly");
  }

  loadEncyclopedia();
  setupBasketballDragEvents();
});
