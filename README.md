# Driver Monitoring System (DMS)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-Runtime-005CED?logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Driver Monitoring System (DMS)** is a real-time driver monitoring system based on Computer Vision and Deep Learning. It detects faces, estimates head pose, analyzes facial attributes (eyes, glasses, mask, etc.), and raises alerts when the driver shows signs of **drowsiness**, **distraction**, or **not watching the road**.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Architectural Overview](#2-architectural-overview)
3. [System Context and Scope](#3-system-context-and-scope)
4. [System Components](#4-system-components)
5. [Data Flow](#5-data-flow)
6. [Processing Workflow](#6-processing-workflow)
7. [Physical Architecture](#7-physical-architecture)
8. [Technology Stack](#8-technology-stack)
9. [Detailed Component Design](#9-detailed-component-design)
10. [Project Structure](#10-project-structure)
11. [System Requirements](#11-system-requirements)
12. [Installation](#12-installation)
13. [Configuration](#13-configuration)
14. [Usage](#14-usage)
15. [Testing](#15-testing)
16. [Roadmap](#16-roadmap)
17. [Contributing](#17-contributing)
18. [License](#18-license)
19. [Acknowledgements](#19-acknowledgements)

---

## 1. Introduction

### 1.1 Purpose

This document describes the complete software architecture of the Driver Monitoring System (DMS). It provides a comprehensive view of the system structure, components, interfaces, data flow, and deployment model to guide implementation and serve as a reference document for an undergraduate capstone project.

### 1.2 Scope

The architecture covers:

- System components and their interactions
- Hardware and software deployment structure
- Data processing pipelines and algorithms
- Technology stack and frameworks used
- Interface specifications between components
- Database schema and data management
- Security and performance considerations

### 1.3 Architectural Goals

| Goal | Description | Priority |
|---|---|---|
| **Real-time performance** | Process video at ≥15 FPS with alert latency <200 ms | MUST |
| **Modularity** | Independent, low-coupling components for maintainability | MUST |
| **Accuracy** | Drowsiness detection ≥90%, activity recognition ≥85% | MUST |
| **Extensibility** | Easy to add new detection features | SHOULD |
| **Resource efficiency** | Run on embedded devices (Jetson Nano / RPi 4) | MUST |
| **Reliability** | Maintain stable operation when subsidiary components fail | SHOULD |

### 1.4 Architectural Principles

1. **Separation of Concerns**: Detection, Analysis, and Action layers operate independently.
2. **Single Responsibility**: Each component has one primary function.
3. **Pipeline Pattern**: Sequential processing stages with clearly defined data contracts.
4. **Event-Driven Alerts**: Alerts are generated asynchronously based on analysis results.
5. **Configuration over Code**: Thresholds and parameters are stored in config files.
6. **Fail-Safe Design**: The system continues to operate even when non-critical components fail.

---

## 2. Architectural Overview

### 2.1 Architectural Style

**Primary pattern:** **Pipeline Architecture** (Pipes and Filters)

The system follows a **linear pipeline architecture** in which video frames flow through sequential processing stages:

```
Video Input → Detection Stage → Analysis Stage → Action Stage → Output
```

**Reasoning:**

- Naturally fits a video processing workflow.
- Clear data transformation at each stage.
- Easy to optimize each stage individually.
- Simple to understand and maintain.
- Well suited for real-time streaming data.

**Supporting patterns:**

- **Layered Architecture**: Detection, Analysis, Action layers.
- **Repository Pattern**: Centralized data storage (SQLite database).
- **Observer Pattern**: Components subscribe to alert events.

### 2.2 High-Level Architecture Diagram

```diagram
╭─────────────────────────────────────────────────────────────────╮
│                   DRIVER MONITORING SYSTEM                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│   ╭─────────────╮         ╭──────────────────────────────────╮ │
│   │  IR Camera  │────────▶│           INPUT LAYER             │ │
│   │   (Video)   │         │  - Frame acquisition (OpenCV)     │ │
│   ╰─────────────╯         │  - IR image preprocessing         │ │
│                           ╰────────────────┬──────────────────╯ │
│                                            │                     │
│                                            ▼                     │
│   ╭─────────────────────────────────────────────────────────╮ │
│   │              DETECTION LAYER (Computer Vision)           │ │
│   │  ╭───────────╮ ╭───────────╮ ╭────────────╮             │ │
│   │  │   Face    │ │    Eye    │ │ Head Pose  │             │ │
│   │  │ Detection │ │  Tracker  │ │ Estimation │             │ │
│   │  ╰───────────╯ ╰───────────╯ ╰────────────╯             │ │
│   │  ╭───────────╮ ╭───────────╮ ╭────────────╮             │ │
│   │  │   Gaze    │ │   Mouth   │ │    Hand    │             │ │
│   │  │  Tracker  │ │ Detection │ │ Detection  │             │ │
│   │  ╰───────────╯ ╰───────────╯ ╰────────────╯             │ │
│   │  ╭───────────╮ ╭────────────╮                           │ │
│   │  │ Activity  │ │ Passenger  │                           │ │
│   │  │Recognition│ │  Counter   │                           │ │
│   │  ╰───────────╯ ╰────────────╯                           │ │
│   ╰────────────────────────┬────────────────────────────────╯ │
│                            ▼                                     │
│   ╭─────────────────────────────────────────────────────────╮ │
│   │            ANALYSIS LAYER (AI Decision Making)           │ │
│   │  ╭────────────╮ ╭────────────╮ ╭────────────╮           │ │
│   │  │ Drowsiness │ │Distraction │ │  Activity  │           │ │
│   │  │  Analyzer  │ │  Analyzer  │ │  Analyzer  │           │ │
│   │  ╰────────────╯ ╰────────────╯ ╰────────────╯           │ │
│   │           ╭───────────────────────╮                      │ │
│   │           │  Risk Assessment      │                      │ │
│   │           │  Engine (aggregator)  │                      │ │
│   │           ╰───────────────────────╯                      │ │
│   ╰────────────────────────┬────────────────────────────────╯ │
│                            ▼                                     │
│   ╭─────────────────────────────────────────────────────────╮ │
│   │            ACTION LAYER (Response & Storage)             │ │
│   │  ╭────────────╮ ╭────────────╮ ╭────────────╮           │ │
│   │  │   Alert    │ │   Event    │ │   Report   │           │ │
│   │  │  Manager   │ │   Logger   │ │  Generator │           │ │
│   │  ╰────────────╯ ╰────────────╯ ╰────────────╯           │ │
│   ╰────────────────────────┬────────────────────────────────╯ │
│                            ▼                                     │
│   ╭─────────────────────────────────────────────────────────╮ │
│   │                     OUTPUT LAYER                         │ │
│   │  - Audio / Visual alerts (speaker, LED)                 │ │
│   │  - Dashboard UI (optional monitor)                      │ │
│   │  - Database storage (SQLite)                            │ │
│   │  - Log files (system and event)                         │ │
│   ╰─────────────────────────────────────────────────────────╯ │
╰─────────────────────────────────────────────────────────────────╯
```

### 2.3 Layer Responsibilities

| Layer | Responsibility | Input | Output |
|---|---|---|---|
| **Input** | Acquire and preprocess video frames | IR camera stream | Preprocessed frame (640x480, grayscale) |
| **Detection** | Extract features using CV | Preprocessed frame | Feature vector (landmarks, eye state, pose) |
| **Analysis** | Interpret features to assess driver state | Feature vector | Analysis results (drowsiness level, distraction score) |
| **Action** | Generate alerts and log events | Analysis results | Alerts, DB records, logs |
| **Output** | Present information to the driver/system | Alerts, logs | Audio/visual alerts, UI, stored data |

---

## 3. System Context and Scope

### 3.1 Context Diagram

```diagram
                  ╭─────────────────────────────────╮
                  │       EXTERNAL ENVIRONMENT       │
                  │                                  │
                  │  ╭────────╮       ╭───────────╮ │
                  │  │ Driver │       │  Vehicle  │ │
                  │  │(human) │       │  Interior │ │
                  │  ╰───┬────╯       ╰─────┬─────╯ │
                  │      │ observed by      │       │
                  │      │                  │ in    │
                  ╰──────┼──────────────────┼───────╯
                         ▼                  ▼
╭───────────────────────────────────────────────────────╮
│            DRIVER MONITORING SYSTEM (scope)            │
│                                                         │
│   ╭──────────────╮       ╭──────────────────╮         │
│   │  IR Camera   │──────▶│ Main Processor   │         │
│   ╰──────────────╯       ╰────────┬─────────╯         │
│                                   │                    │
│                          ╭────────▼────────╮           │
│                          │   Data Storage  │           │
│                          │     (SQLite)    │           │
│                          ╰─────────────────╯           │
│                                   │                    │
│   ╭──────────────╮       ╭────────▼────────╮          │
│   │ Audio/Visual │◀──────│  Alert Output   │          │
│   │   Alerts     │       │                 │          │
│   ╰──────────────╯       ╰─────────────────╯          │
╰───────────────────────────────────────────────────────╯
                            │
                            ▼
                  ╭─────────────────────╮
                  │   External Users    │
                  │  (fleet managers,   │
                  │   log reviewers)    │
                  ╰─────────────────────╯
```

### 3.2 In Scope

- Video acquisition and image processing.
- Face, eye, and activity detection.
- Driver state analysis (drowsiness, distraction).
- Alert generation and delivery.
- Event logging and data storage.
- System health monitoring.

### 3.3 Out of Scope

- Vehicle control systems (brake, steering).
- Internet connectivity / cloud services.
- Advanced Driver Assistance Systems (ADAS).
- In-vehicle infotainment integration.
- Multi-camera setups (single IR camera only).
- Driver authentication system (left for future research).

### 3.4 External Interfaces

| Interface | Type | Direction | Description |
|---|---|---|---|
| **IR Camera** | Hardware | Input | USB-connected IR camera providing 30 fps video stream |
| **Speaker** | Hardware | Output | USB / 3.5mm audio output for alert tones |
| **LED Indicator** | Hardware | Output | GPIO-controlled LED for visual alerts |
| **Power Supply** | Hardware | Input | 5V USB-C source (for Jetson Nano / RPi 4) |
| **File System** | Software | Bi-directional | Local storage for DB, logs, config files |
| **Operating System** | Software | Bi-directional | Linux (Ubuntu 20.04 / Raspberry Pi OS) |

---

## 4. System Components

### 4.1 Component Diagram

```diagram
╭─────────────────────────────────────────────────────────────────╮
│                  DRIVER MONITORING SYSTEM                        │
│                                                                   │
│   ╭───────────────────────────────────────────────────────────╮ │
│   │ INPUT SUBSYSTEM                                            │ │
│   │  ╭────────────────╮      ╭────────────────╮               │ │
│   │  │ Video Capture  │─────▶│  Preprocessor  │               │ │
│   │  ╰────────────────╯      ╰────────────────╯               │ │
│   ╰────────────────────────────┬───────────────────────────────╯ │
│                                │ frames                           │
│   ╭────────────────────────────▼───────────────────────────────╮ │
│   │ DETECTION SUBSYSTEM                                         │ │
│   │  ╭──────────────╮ ╭──────────────╮ ╭──────────────╮       │ │
│   │  │Face Detector │ │ Eye Tracker  │ │  Head Pose   │       │ │
│   │  ╰──────────────╯ ╰──────────────╯ ╰──────────────╯       │ │
│   │  ╭──────────────╮ ╭──────────────╮ ╭──────────────╮       │ │
│   │  │ Gaze Tracker │ │Mouth Detector│ │Hand Detector │       │ │
│   │  ╰──────────────╯ ╰──────────────╯ ╰──────────────╯       │ │
│   │  ╭──────────────╮ ╭──────────────╮                        │ │
│   │  │   Activity   │ │  Passenger   │                        │ │
│   │  │  Recognizer  │ │   Counter    │                        │ │
│   │  ╰──────────────╯ ╰──────────────╯                        │ │
│   ╰─────────────────────────────┬──────────────────────────────╯ │
│                                 │ features                         │
│   ╭─────────────────────────────▼──────────────────────────────╮ │
│   │ ANALYSIS SUBSYSTEM                                           │ │
│   │  ╭──────────────╮ ╭──────────────╮ ╭──────────────╮        │ │
│   │  │  Drowsiness  │ │ Distraction  │ │   Activity   │        │ │
│   │  │   Analyzer   │ │   Analyzer   │ │   Analyzer   │        │ │
│   │  ╰──────┬───────╯ ╰──────┬───────╯ ╰──────┬───────╯        │ │
│   │         └────────────────┼────────────────┘                  │ │
│   │                          ▼                                    │ │
│   │              ╭─────────────────────╮                         │ │
│   │              │ Risk Assessment Eng.│                         │ │
│   │              ╰─────────────────────╯                         │ │
│   ╰─────────────────────────────┬──────────────────────────────╯ │
│                                 │ risk events                     │
│   ╭─────────────────────────────▼──────────────────────────────╮ │
│   │ ACTION SUBSYSTEM                                             │ │
│   │  ╭──────────────╮ ╭──────────────╮ ╭──────────────╮        │ │
│   │  │Alert Manager │ │ Event Logger │ │Report Gener. │        │ │
│   │  ╰──────────────╯ ╰──────────────╯ ╰──────────────╯        │ │
│   ╰─────────────────────────────┬──────────────────────────────╯ │
│                                 ▼                                 │
│   ╭────────────────────────────────────────────────────────────╮ │
│   │ STORAGE SUBSYSTEM                                            │ │
│   │  ╭──────────────╮  ╭──────────────╮                         │ │
│   │  │ DB Manager   │  │File Storage  │                         │ │
│   │  ╰──────────────╯  ╰──────────────╯                         │ │
│   ╰────────────────────────────────────────────────────────────╯ │
│                                                                   │
│   ╭────────────────────────────────────────────────────────────╮ │
│   │ SUPPORT SUBSYSTEM                                            │ │
│   │  ╭──────────────╮ ╭──────────────╮ ╭──────────────╮        │ │
│   │  │ Config Mgr   │ │ Sys Monitor  │ │   Logger     │        │ │
│   │  ╰──────────────╯ ╰──────────────╯ ╰──────────────╯        │ │
│   ╰────────────────────────────────────────────────────────────╯ │
╰─────────────────────────────────────────────────────────────────╯
```

### 4.2 Component Responsibilities

#### Input Subsystem

| Component | Responsibility | Key Functions |
|---|---|---|
| **Video Capture** | Interface with the IR camera, capture frames | `init_camera()`, `get_frame()`, `release()` |
| **Preprocessor** | Prepare frames (resize, normalize, enhance) | `preprocess()`, `enhance_ir_image()`, `normalize()` |

#### Detection Subsystem

| Component | Responsibility | Key Functions |
|---|---|---|
| **Face Detector** | Detect face and extract 68 landmarks | `detect_face()`, `get_landmarks()` |
| **Eye Tracker** | Track eyes, compute Eye Aspect Ratio (EAR) | `track_eyes()`, `compute_ear()`, `detect_blink()` |
| **Head Pose Estimator** | Estimate 3D head pose (pitch, yaw, roll) | `estimate_pose()`, `get_rotation_angles()` |
| **Gaze Tracker** | Track gaze direction and focus | `track_gaze()`, `compute_gaze_vector()` |
| **Mouth Detector** | Detect mouth state (open/closed, yawning) | `detect_mouth()`, `compute_mar()` |
| **Hand Detector** | Detect hand position (on-wheel, off-wheel, holding phone) | `detect_hands()`, `classify_hand_activity()` |
| **Activity Recognizer** | Recognize hazardous actions (calling, drinking, smoking, etc.) | `recognize_activity()`, `classify_object()` |
| **Passenger Counter** | Count people in the vehicle | `count_passengers()`, `detect_multiple_faces()` |

#### Analysis Subsystem

| Component | Responsibility | Key Functions |
|---|---|---|
| **Drowsiness Analyzer** | Analyze drowsiness level from eye, mouth, head data | `analyze_drowsiness()`, `compute_perclos()`, `detect_microsleep()` |
| **Distraction Analyzer** | Analyze distraction from gaze and head pose | `analyze_distraction()`, `compute_attention_score()` |
| **Activity Analyzer** | Analyze hazardous activities and assess risk | `analyze_activity()`, `assess_activity_risk()` |
| **Risk Assessment Engine** | Aggregate analysis results, compute overall risk score | `assess_overall_risk()`, `generate_event()` |

#### Action Subsystem

| Component | Responsibility | Key Functions |
|---|---|---|
| **Alert Manager** | Generate and deliver audio/visual alerts | `trigger_alert()`, `play_audio()`, `blink_led()` |
| **Event Logger** | Log events into database and files | `log_event()`, `store_to_db()`, `write_to_file()` |
| **Report Generator** | Produce summary reports and analytics | `generate_trip_report()`, `create_statistics()` |

#### Storage Subsystem

| Component | Responsibility | Key Functions |
|---|---|---|
| **Database Manager** | Manage SQLite connections and queries | `connect()`, `execute_query()`, `insert_event()` |
| **File Storage Manager** | Manage log and configuration files | `write_log()`, `read_config()`, `save_snapshot()` |

#### Support Subsystem

| Component | Responsibility | Key Functions |
|---|---|---|
| **Config Manager** | Load and manage system configuration | `load_config()`, `get_threshold()`, `update_setting()` |
| **System Monitor** | Monitor system health (CPU, memory, FPS) | `monitor_resources()`, `check_health()`, `log_performance()` |
| **Logger** | Centralized logging for debugging | `log_info()`, `log_error()`, `log_debug()` |

---

## 5. Data Flow

### 5.1 Main Processing Pipeline

```diagram
╭────────────╮
│ IR Camera  │
│  (30 FPS)  │
╰─────┬──────╯
      │ raw frame (640x480 RGB)
      ▼
╭───────────────────╮
│  Video Capture    │
│  - capture frame  │
│  - validate       │
╰─────────┬─────────╯
          │ frame: np.ndarray(480,640,3)
          ▼
╭───────────────────╮
│  Preprocessor     │
│  - resize         │
│  - to grayscale   │
│  - normalize      │
│  - enhance (CLAHE)│
╰─────────┬─────────╯
          │ preprocessed_frame
          ▼
╭───────────────────────────────────────────────╮
│        DETECTION STAGE (parallelizable)        │
│                                                 │
│   ╭─────────────╮      ╭──────────────╮       │
│   │  Face Det.  │─────▶│ Landmarks(68)│       │
│   ╰─────────────╯      ╰──────┬───────╯       │
│                               │                │
│           ╭───────────────────┴──────────╮     │
│           ▼                              ▼     │
│   ╭─────────────╮                ╭────────────╮│
│   │ Eye Tracker │                │Mouth Detect││
│   ╰─────────────╯                ╰────────────╯│
│           │                              │     │
│           ▼                              ▼     │
│   ╭─────────────╮                ╭────────────╮│
│   │ Head Pose   │                │  Activity  ││
│   │ + Gaze      │                │ Recognizer ││
│   ╰─────────────╯                ╰────────────╯│
╰────────────────────┬───────────────────────────╯
                     │ DetectionData {
                     │   face_landmarks, ear,
                     │   mar, head_pose, gaze,
                     │   hands_on_wheel,
                     │   detected_activity,
                     │   timestamp
                     │ }
                     ▼
╭───────────────────────────────────────────────╮
│            ANALYSIS STAGE                      │
│                                                 │
│   ╭──────────────╮  ╭──────────────╮          │
│   │  Drowsiness  │  │ Distraction  │          │
│   │   Analyzer   │  │   Analyzer   │          │
│   ╰──────┬───────╯  ╰──────┬───────╯          │
│          │                 │                   │
│          ▼                 ▼                   │
│      ╭─────────────────────────╮               │
│      │  Risk Assessment Engine │               │
│      ╰─────────────────────────╯               │
╰────────────────────┬───────────────────────────╯
                     │ RiskEvent {
                     │   event_type, severity,
                     │   risk_score,
                     │   drowsiness_level,
                     │   attention_score,
                     │   requires_alert,
                     │   timestamp
                     │ }
                     ▼
╭───────────────────────────────────────────────╮
│             ACTION STAGE                       │
│   1. Alert Manager (audio + LED, async)        │
│   2. Event Logger  (DB + file, async)          │
│   3. Report Generator (on shutdown)            │
╰────────────────────┬───────────────────────────╯
                     ▼
              ╭───────────────╮
              │   OUTPUTS:    │
              │  audio alerts │
              │  LED blinking │
              │  DB records   │
              │  log files    │
              ╰───────────────╯
```

### 5.2 Per-Frame Timing Budget

Target: < 67 ms per frame (≥15 FPS).

| Stage | Component | Estimated Time | Cumulative |
|---|---|---|---|
| 1 | Video Capture | 5 ms | 5 ms |
| 2 | Preprocessing | 3 ms | 8 ms |
| 3 | Face Detection | 15 ms | 23 ms |
| 4 | Eye + Mouth Tracking | 8 ms | 31 ms |
| 5 | Head Pose + Gaze | 10 ms | 41 ms |
| 6 | Hand + Activity Recognition | 12 ms | 53 ms |
| 7 | Drowsiness + Distraction + Activity Analysis | 8 ms | 61 ms |
| 8 | Risk Assessment | 2 ms | 63 ms |
| 9 | Alerts + Logging | 3 ms | 66 ms |
| **Total** | | **66 ms** | **≈ 15 FPS** |

**Performance notes:**

- Detection stage (steps 3–6) can be partially parallelized on the Jetson Nano GPU.
- Alerts and logging (step 9) run asynchronously to avoid blocking the next frame.
- If processing exceeds 67 ms, frames are dropped (graceful degradation).

---

## 6. Processing Workflow

### 6.1 Main Process Flow

```diagram
╭───────────────────────────────────────────────╮
│              SYSTEM STARTUP                    │
╰────────────────┬──────────────────────────────╯
                 ▼
        ╭────────────────╮
        │  Load Config   │
        ╰────────┬───────╯
                 ▼
        ╭────────────────╮
        │ Init Database  │
        ╰────────┬───────╯
                 ▼
        ╭────────────────╮
        │  Load Models   │
        ╰────────┬───────╯
                 ▼
        ╭────────────────╮
        │  Init Camera   │
        ╰────────┬───────╯
                 ▼
        ╭────────────────╮
        │ Init Hardware  │
        │ (speaker/LED)  │
        ╰────────┬───────╯
                 ▼
╭───────────────────────────────────────────────╮
│           MAIN PROCESSING LOOP                 │
│  while running:                                │
│    1. Capture frame                            │
│    2. Preprocess frame                         │
│    3. Detection stage                          │
│       a. Face + landmarks                      │
│       b. Eye tracking (EAR)                    │
│       c. Mouth (MAR)                           │
│       d. Head pose                             │
│       e. Gaze                                  │
│       f. Hand + activity                       │
│    4. Analysis stage                           │
│       a. Drowsiness (PERCLOS, microsleep)      │
│       b. Distraction (gaze, head yaw)          │
│       c. Activity risk                         │
│    5. Risk assessment                          │
│    6. Trigger alert (async)                    │
│    7. Log event (async)                        │
│    8. Monitor system (FPS, CPU, memory)        │
│    9. Loop control (check shutdown signal)     │
╰────────────────┬──────────────────────────────╯
                 ▼ (on shutdown)
╭───────────────────────────────────────────────╮
│              SYSTEM SHUTDOWN                   │
│  - Release camera                              │
│  - Close database                              │
│  - Generate trip report                        │
│  - Cleanup GPIO                                │
│  - Exit                                        │
╰───────────────────────────────────────────────╯
```

### 6.2 Driver State Machine

```diagram
                    ╭─────────╮
                    │  START  │
                    ╰────┬────╯
                         ▼
              ╭───────────────────╮
         ┌───▶│     NO FACE       │◀───┐
         │    │ (face not visible)│    │
         │    ╰────────┬──────────╯    │
         │  face seen  │               │ face lost > 3s
         │             ▼               │
         │    ╭───────────────────╮    │
         │    │      NORMAL       │────┘
         │    │ (alert, focused)  │
         │    ╰────────┬──────────╯
         │             │ drowsiness signals
         │             ▼
         │    ╭───────────────────╮
         │    │   MILD DROWSY     │
         │    │ (low EAR, MAR ↑)  │
         │    ╰────────┬──────────╯
         │             │ persistent > 5s
         │             ▼
         │    ╭───────────────────╮
         └────│  SEVERE DROWSY    │
              │ (PERCLOS>80%,     │
              │  microsleep)      │
              ╰───────────────────╯
                       │ recovers
                       ▼
                  back to NORMAL

         ╭───────────────────╮
    ┌───▶│    DISTRACTED     │◀──────┐
    │    │ (looking away)    │       │
    │    ╰───────────────────╯       │
    │             │ refocus           │ look-away > 2s
    │             ▼                   │
    │    ╭───────────────────╮        │
    └────│      NORMAL       │────────┘
         ╰────────┬──────────╯
                  │ hazardous activity
                  ▼
         ╭───────────────────╮
         │ HAZARDOUS ACTION  │
         │ (calling, eating, │
         │  smoking, etc.)   │
         ╰───────────────────╯
                  │ stops activity
                  ▼
             back to NORMAL
```

---

## 7. Physical Architecture

### 7.1 Hardware Layout

```diagram
╭───────────────────────────────────────────────────────╮
│                  PHYSICAL SYSTEM LAYOUT                │
│                                                         │
│   ╭─────────────────────╮                              │
│   │     IR Camera        │                              │
│   │  - 640x480 / 30 fps  │                              │
│   │  - 850nm IR LEDs     │                              │
│   │  - USB 2.0           │                              │
│   ╰──────────┬──────────╯                              │
│              │ USB cable (1.5 m)                        │
│              ▼                                          │
│   ╭───────────────────────────────────────────────╮   │
│   │ Edge Computing Device                          │   │
│   │ (Jetson Nano 4GB / Raspberry Pi 4 8GB)         │   │
│   │ Connectors:                                    │   │
│   │  - USB 2.0/3.0 x4 (camera, speaker, periph.)  │   │
│   │  - 40-pin GPIO header (LED)                    │   │
│   │  - microSD slot (storage)                      │   │
│   │  - Power input (USB-C or DC)                   │   │
│   ╰────────────┬───────────┬───────────┬───────────╯   │
│                ▼           ▼           ▼               │
│           ╭────────╮  ╭────────╮ ╭───────────╮        │
│           │ LED   │  │ Speaker│ │  Power    │        │
│           │ strip │  │ (USB / │ │  5V/4A    │        │
│           │(GPIO) │  │ 3.5mm) │ │  USB-C    │        │
│           ╰────────╯  ╰────────╯ ╰───────────╯        │
╰───────────────────────────────────────────────────────╯
```

### 7.2 Hardware Options

#### Option 1: NVIDIA Jetson Nano 4GB (Recommended)

| Component | Specification | Purpose |
|---|---|---|
| **CPU** | Quad-core ARM Cortex-A57 @ 1.43 GHz | Main processing |
| **GPU** | 128-core NVIDIA Maxwell | Inference acceleration |
| **RAM** | 4GB LPDDR4 | Application memory |
| **Storage** | microSD 64GB | OS, app, DB, logs |
| **Camera I/F** | USB 2.0 | IR camera connection |
| **Audio Out** | 3.5mm / HDMI | Alert audio |
| **GPIO** | 40-pin header | LED indicators |
| **Power** | 5V / 4A (USB-C / DC) | System power |
| **Cost** | ~$100-120 | |

**Pros:** GPU acceleration for TensorFlow models, better real-time performance (20–25 FPS expected), headroom for future features.

#### Option 2: Raspberry Pi 4 Model B 8GB (Alternative)

| Component | Specification | Purpose |
|---|---|---|
| **CPU** | Quad-core ARM Cortex-A72 @ 1.5 GHz | Main processing |
| **RAM** | 8GB LPDDR4 | Application memory |
| **Storage** | microSD 64GB | OS, app, DB, logs |
| **Camera I/F** | USB 3.0 | IR camera connection |
| **Power** | 5V / 3A (USB-C) | System power |
| **Cost** | ~$75-90 | |

**Pros:** Lower cost, larger RAM (8GB), easier procurement.
**Cons:** No GPU acceleration (CPU-only inference), lower FPS (15–18 FPS expected).

### 7.3 In-Vehicle Installation

**Installation requirements:**

1. **Camera location**: Center of dashboard, 20–40 cm from the driver's face, tilted 10–15° downward.
2. **LED location**: Within driver's peripheral vision, not blocking road view.
3. **Device mounting**: Securely mounted to prevent movement/vibration.
4. **Cable management**: All cables fixed and never tangled with steering controls.
5. **Power connection**: Stable 5V supply via a quality 12V→5V converter.

---

## 8. Technology Stack

### 8.1 Stack Overview

```diagram
╭─────────────────────────────────────────────────────╮
│                  APPLICATION LAYER                   │
│   Driver Monitoring System (Python 3.9+)            │
│   - Custom detection / analysis / action modules    │
╰────────────────────────┬────────────────────────────╯
                         ▼
╭─────────────────────────────────────────────────────╮
│            FRAMEWORKS AND LIBRARIES                  │
│   Computer Vision:                                   │
│    - OpenCV 4.x                                      │
│    - dlib 19.22 (face detection, landmarks)         │
│   Machine Learning:                                  │
│    - PyTorch 2.0+ (head pose, attribute models)     │
│    - ONNX Runtime (face detector inference)         │
│    - NumPy, SciPy                                    │
│   Data and Storage:                                  │
│    - SQLite3                                         │
│    - PyYAML                                          │
│   Hardware Interface:                                │
│    - RPi.GPIO / Jetson.GPIO                         │
│    - pygame (audio playback)                        │
╰────────────────────────┬────────────────────────────╯
                         ▼
╭─────────────────────────────────────────────────────╮
│                 OPERATING SYSTEM                     │
│  - Ubuntu 20.04 LTS / Raspberry Pi OS / Windows     │
╰────────────────────────┬────────────────────────────╯
                         ▼
╭─────────────────────────────────────────────────────╮
│                   HARDWARE LAYER                     │
│  Jetson Nano / RPi 4 / IR Camera / Speaker / LED    │
╰─────────────────────────────────────────────────────╯
```

### 8.2 Technology Choices

| Category | Technology | Version | Rationale |
|---|---|---|---|
| **Language** | Python | 3.9+ | Excellent CV/ML library support, fast prototyping |
| **CV** | OpenCV | 4.x | Industry standard, hardware-accelerated, real-time ready |
| **Face landmarks** | dlib | 19.22 | High accuracy, pretrained 68-landmark model |
| **DL Inference** | ONNX Runtime | latest | Optimized cross-platform inference for face detector |
| **DL Framework** | PyTorch | 2.0+ | Head pose (MobileNetV2), facial attributes |
| **Numerics** | NumPy / SciPy | 1.21 / 1.7 | Array ops, signal smoothing |
| **Database** | SQLite3 | 3.x | Embedded, ACID, single-user friendly |
| **Config** | PyYAML | 6.0+ | Human-readable, easy threshold tuning |
| **Audio** | pygame | 2.1 | Simple audio API, cross-platform |
| **GPIO** | RPi/Jetson.GPIO | latest | Official LED control APIs |

### 8.3 Pretrained Models

| Model | Source | Purpose | Size | Format |
|---|---|---|---|---|
| **det_2.5g.onnx** | InsightFace | Face detection | ~1.5 MB | `.onnx` |
| **mobilenetv2.pt** | Custom training | Head pose estimation (yaw/pitch/roll) | 12 MB | `.pt` |
| **face_landmarker.task** | MediaPipe | 3DMM face landmarks (68/106 pts) | ~3 MB | `.task` |
| **FaceAttribNet** | Custom | Eye/glasses/mask attribute classification | ~5 MB | `.pt` |

---

## 9. Detailed Component Design

### 9.1 Face Detector

**Responsibility:** Detect the driver's face and provide bounding boxes for downstream landmark/attribute models.

**Algorithm:** Lightweight ONNX face detector (`det_2.5g.onnx`).

**Key parameters:**

- `confidence_threshold = 0.5`
- `nms_threshold = 0.4`
- `no_face_threshold = 30` frames (about 1 second at 30 fps)

### 9.2 Eye Tracker (EAR-based)

**Responsibility:** Track eyes and compute the Eye Aspect Ratio (EAR) for blink and closed-eye detection.

**EAR formula:**

```
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
```

where `p1..p6` are the six eye landmarks.

**Reference values:**

- Open eye: EAR ≈ 0.30 – 0.40
- Half-closed: EAR ≈ 0.15 – 0.25
- Closed: EAR ≈ 0.05 – 0.15

**Thresholds:**

- `ear_threshold = 0.25` — below this means eyes closed.
- `blink_threshold = 3` consecutive low-EAR frames count as one blink.

### 9.3 Head Pose Estimator

**Responsibility:** Estimate 3D head pose (pitch, yaw, roll).

**Algorithm:** MobileNetV2 regression network (`mobilenetv2.pt`) or `cv2.solvePnP` using a 3D head model and 2D landmark correspondences.

**Angle interpretation:**

- **Pitch** (X axis): + down / − up. Drowsy if > +20°.
- **Yaw** (Y axis): + right / − left. Distracted if |yaw| > 30°.
- **Roll** (Z axis): + tilt right / − tilt left. Normal range −10° to +10°.

### 9.4 Gaze Tracker

**Responsibility:** Track gaze direction based on iris position relative to the eye corners.

**Output:**

- `gaze_x ∈ [-1, +1]`: −1 far left, 0 forward, +1 far right.
- `gaze_y ∈ [-1, +1]`: −1 up, 0 forward, +1 down.

**Look-away threshold:** `|gaze_x| > 0.4` or `|gaze_y| > 0.4`.

### 9.5 Drowsiness Analyzer

**Responsibility:** Compute multi-factor drowsiness score from EAR, MAR, and head pitch.

**Drowsiness contributions:**

| Indicator | Threshold | Weight |
|---|---|---|
| PERCLOS > 80% | Eyes closed > 80% of recent frames | +40 |
| PERCLOS > 50% | Eyes closed > 50% of recent frames | +20 |
| Yawn detected | MAR > 0.6 | +20 |
| Microsleep | Eyes closed > 1 s | +30 |
| Head down | Pitch > 20° | +10 |

**Severity scale:**

- 0–30: Alert (green)
- 31–59: Mild drowsiness (yellow)
- 60–100: Drowsy — alert immediately (red)

### 9.6 Risk Assessment Engine

Aggregates outputs from all analyzers:

```
risk_score = w1 * drowsiness_score
           + w2 * distraction_score
           + w3 * activity_risk_score
```

The engine emits a `RiskEvent` with `event_type`, `severity` (LOW / MEDIUM / HIGH), and a `requires_alert` flag.

---

## 10. Project Structure

```
driver_monitoring/
|-- main.py                     # Thin executable entry point
|-- app/                        # Bootstrap, lifecycle, dependency assembly
|-- pipeline/                   # Shared frame context and pipeline runner
|-- features/                   # Feature-first detection and analysis modules
|   |-- face/                   # Face and facial-attribute detection
|   |-- landmarks/              # Landmark detection and constants
|   |-- head_pose/              # Head-pose estimation and feedback
|   |-- gaze/                   # Eye-gaze estimation and telemetry
|   |-- drowsiness/             # Drowsiness analysis
|   |-- distraction/            # Distraction analysis
|   `-- risk/                   # Risk events and aggregation
|-- infrastructure/             # Camera, database, preprocessing, event logging
|-- presentation/opencv/        # OpenCV visualization and debug stages
|-- alerting/                    # Risk-event alert handlers
|-- utils/                      # Reusable model and image helpers
|-- models/                     # Model weights
|-- data/                       # Test media and local database
|-- logs/                       # Runtime logs
|-- tests/                      # Automated and diagnostic tests
|-- scripts/                    # Developer utilities
|-- config.yaml                 # Camera, log, and module toggles
`-- requirements.txt            # Python dependencies
```

Dependency flow is kept one-way: `main -> app -> pipeline/features ->
infrastructure/presentation/alerting`. Feature-specific code stays inside its
feature package; only genuinely shared contracts and helpers belong in
`pipeline/` or `utils/`.

---

## 11. System Requirements

- Python **3.9+**
- (Recommended) NVIDIA GPU with CUDA for faster PyTorch / ONNX Runtime inference
- Operating System: **Windows / Linux / macOS**

### Main Dependencies

```text
onnx
onnxruntime          # use onnxruntime-gpu instead if CUDA is available
opencv-python
numpy
scikit-image
faiss-cpu
PyYAML>=6.0.2
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
```

---

## 12. Installation

### 12.1 Clone the repository

```bash
git clone https://github.com/thanhhung-dev/driver_monitoring.git
cd driver_monitoring
```

### 12.2 Create a virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 12.3 Install dependencies

```bash
pip install -r requirements.txt
```

> If you have an NVIDIA GPU, install `onnxruntime-gpu` instead of `onnxruntime`, and the matching PyTorch CUDA build.

### 12.4 Download model weights

Place the following files into `models/`:

| File | Description |
|---|---|
| `det_2.5g.onnx` | Face detector (required) |
| `mobilenetv2.pt` | Head pose estimation |
| `face_landmarker.task` | MediaPipe face landmarker |

---

## 13. Configuration

Configuration is stored in `config.yaml`:

```yaml
camera:
  source: "data/dataset.mp4"   # video path, or use an integer (0, 1, ...) for a webcam
  device_id: 0
  resolution:
    width: 640
    height: 480
  fps: 15
  frame_timeout_ms: 2000

system:
  log_level: INFO
  log_file: logs/dms.log

# Enable/disable each model in the pipeline.
# face_detector is always required.
models:
  facemap: false       # FaceMap 3DMM landmarks
  attrib: false        # Facial attribute (eyes/glasses/mask...)
  head_pose: true      # MobileNetV2 head pose
  analyzer: false      # Drowsiness analyzer
  visualizer: true     # OpenCV display window
```

---

## 14. Usage

```bash
python main.py
```

- The OpenCV window shows the video with bounding boxes, landmarks, and alerts (when `visualizer` is enabled).
- Press **`q`** (or `Ctrl+C` in the terminal) to quit.
- Logs are written to `logs/dms.log`.

---

## 15. Testing

```bash
python -m pytest tests/
```

---

## 16. Roadmap

- [ ] Full **eye gaze tracking** integration.
- [ ] **Audio** alerts when drowsiness is detected.
- [ ] Event publishing via **MQTT / REST API** for in-vehicle systems.
- [ ] Deployment on **Jetson Nano / Raspberry Pi**.
- [ ] Real-time monitoring **web dashboard**.
- [ ] SQLite-based event store and trip report generator.
- [ ] Activity recognition (calling, drinking, smoking, etc.).

---

## 17. Contributing

Contributions are welcome. Please:

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "feat: add feature X"`
4. Push to your fork and open a **Pull Request**

---

## 18. License

Released under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 19. Acknowledgements

This project is built on top of ideas and reference code from:

- [yakhyo/face-reidentification](https://github.com/yakhyo/face-reidentification)
- [yakhyo/uniface](https://github.com/yakhyo/uniface) — all-in-one face analysis library
- [InsightFace](https://github.com/deepinsight/insightface)
- [MediaPipe Face Landmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)

---

<p align="center">
  If you find this project useful, please <b>star</b> the repo to support the author.
</p>
