<div align="center">

# 🌊 PhysTrace
**The Forensic Black-Box for Cyber-Physical Systems.**

✨ *Stop guessing if it's a sensor glitch or a state-sponsored attack. PhysTrace validates industrial telemetry against the immutable laws of physics.* ✨

[![Frontend](https://img.shields.io/badge/Frontend-React_Three_Fiber-0055FF?style=for-the-badge&logo=react)](https://react.dev)
[![Backend](https://img.shields.io/badge/Backend-FastAPI_&_Python-14B8A6?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![Design](https://img.shields.io/badge/UI/UX-Swiss_Engineering-F8F9FA?style=for-the-badge&logo=css3&logoColor=black)](#)

<br/>

[View Demo](#) · [Report Bug](#) · [Request Feature](#)

</div>

---

## ⚡ Overview: The Ultimate ICS Investigation Engine

Traditional dashboards only tell you what the sensors *claim* is happening. **PhysTrace** is a production-ready, mathematically verified investigation workstation designed for Industrial Control Systems (ICS) and operational technology (OT) environments like water-treatment facilities.

Built with an ultra-premium **Swiss Architectural aesthetic**, seamlessly embedded **3D React UI components**, broadcast-quality **GSAP motion graphics**, and a responsive **WebGL Fluid simulation**, PhysTrace merges a multi-million dollar frontend experience with a mathematically rigorous backend forensic engine.

---

## 🏗️ System Architecture

```mermaid
graph TD
    %% Styling Definitions for Swiss Aesthetic
    classDef default fill:#F8F9FA,stroke:#0F172A,stroke-width:1px,color:#0F172A,font-family:monospace;
    classDef container fill:#FFFFFF,stroke:#CBD5E1,stroke-width:1px,color:#0F172A,font-weight:bold;
    classDef frontend fill:#FFFFFF,stroke:#0F172A,stroke-width:2px,color:#0F172A,shadow:true;
    classDef backend fill:#0F172A,stroke:#0F172A,stroke-width:1px,color:#FFFFFF,shadow:true;
    classDef engine fill:#0055FF,stroke:#0055FF,stroke-width:1px,color:#FFFFFF,font-weight:bold,shadow:true;
    classDef security fill:#F59E0B,stroke:#D97706,stroke-width:2px,color:#FFFFFF,font-weight:bold;
    classDef physical fill:#E2E8F0,stroke:#94A3B8,stroke-width:1px,color:#0F172A,stroke-dasharray: 5 5;

    %% Layer 1: Client / Frontend
    subgraph UI ["FRONTEND WORKSTATION (React + Vite)"]
        direction TB
        R3F["3D Rendering Engine\n(React Three Fiber / Drei)"]:::frontend
        GSAP["Motion & Interactions\n(GSAP / Framer Motion)"]:::frontend
        DOM["Swiss UI Overlay\n(Tailwind CSS)"]:::frontend

        R3F --- GSAP
        GSAP --- DOM
    end

    %% Layer 2: API & Backend
    subgraph API ["BACKEND SERVER (FastAPI)"]
        direction LR
        WS["WebSocket Manager\n(Real-Time Telemetry)"]:::backend
        REST["REST API\n(Configuration & Evidence)"]:::backend
        DB[(Local SQLite DB\nFlight Recorder)]:::backend

        WS <--> REST
        REST <--> DB
    end

    %% Layer 3: The Brains
    subgraph CORE ["PHYSTRACE INVESTIGATION ENGINE (Python)"]
        direction TB
        DT["Counterfactual Digital Twin\n(SciPy / NumPy)"]:::engine
        VW["Virtual Witness\n(OpenCV Vision Pipeline)"]:::engine
        WM["Physical Watermarking\n(Actuator Signal Injection)"]:::engine
        RP["Safety-Constrained Response Planner\n(Lockdown Triggers)"]:::security
    end

    %% Layer 4: Cyber-Physical Layer
    subgraph OT ["CYBER-PHYSICAL SYSTEM (TK-101 Water Plant)"]
        direction LR
        PLC["PLC / Network Controller"]:::physical
        ACT["Actuators\n(Pumps / Valves)"]:::physical
        SEN["Digital Sensors\n(Level / Flow)"]:::physical
        CAM["Analog Camera\n(Visual Feed)"]:::physical
    end

    %% Data Flow Routing
    UI <== "Live Data Stream\n(WSS)" ==> WS
    UI <== "HTTP Requests\n(JSON)" ==> REST

    WS <--> CORE
    REST <--> CORE

    PLC ==> |Compromised Data| SEN
    CORE ==> |Watermark Injection| PLC
    CORE -.-> |Hardware Safe-Mode| ACT
    CORE -.-> |Network Isolation| PLC

    SEN ==> |Telemetry| DT
    CAM ==> |Video Stream| VW

    %% Aesthetics
    class UI,API,CORE,OT container;
```