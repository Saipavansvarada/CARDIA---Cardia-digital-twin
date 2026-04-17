import { useEffect, useState } from "react";
import HeartGraph from "./HeartGraph";

function Dashboard() {
  const [data, setData] = useState(null);
  const [connected, setConnected] = useState(false);
  const [lastUpdate, setLastUpdate] = useState(null);

  useEffect(() => {
    const fetchData = () => {
      fetch("http://localhost:8000/live")
        .then((res) => res.json())
        .then((d) => {
          // Only update if real ESP32 data (hr > 0)
          if (d.source === "esp32" && d.heart_rate > 0) {
            setData(d);
            setConnected(true);
            setLastUpdate(new Date().toLocaleTimeString());
          } else if (d.source === "waiting") {
            setConnected(false);
          }
        })
        .catch(() => setConnected(false));
    };

    fetchData();
    const interval = setInterval(fetchData, 2000);
    return () => clearInterval(interval);
  }, []);

  const safetyColor =
    data?.safety === "DANGER"  ? "#ff1744" :
    data?.safety === "CAUTION" ? "#ff9100" : "#00c853";

  const zoneNames  = ["Rest","Zone 1","Zone 2","Zone 3","Zone 4","Zone 5"];
  const zoneColors = ["#90CAF9","#66BB6A","#FFA726","#FF7043","#EF5350","#B71C1C"];

  // ── Waiting screen ─────────────────────────────────────────────────────────
  if (!connected || !data) {
    return (
      <div style={{
        fontFamily: "Arial, sans-serif",
        background: "#0a0a1a",
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        color: "white",
      }}>
        <div style={{ fontSize: "80px", marginBottom: "20px" }}>❤️</div>
        <h1 style={{ fontSize: "32px", marginBottom: "10px" }}>CARDIA</h1>
        <p style={{ color: "#aaa", fontSize: "18px", marginBottom: "30px" }}>
          Waiting for ESP32 sensor...
        </p>
        <div style={{
          background: "#1a1a2e",
          borderRadius: "12px",
          padding: "20px 40px",
          textAlign: "center",
          border: "1px solid #333",
        }}>
          <p style={{ color: "#ff9100", margin: 0 }}>
            👆 Place your finger on the MAX30102 sensor
          </p>
          <p style={{ color: "#666", margin: "10px 0 0", fontSize: "13px" }}>
            Make sure ESP32 is connected to WiFi and backend is running
          </p>
        </div>

        {/* Connection checker */}
        <div style={{ marginTop: "20px", display: "flex", gap: "10px", alignItems: "center" }}>
          <div style={{
            width: "10px", height: "10px",
            borderRadius: "50%",
            background: "#ff1744",
            animation: "pulse 1s infinite",
          }} />
          <span style={{ color: "#666", fontSize: "13px" }}>ESP32 not detected</span>
        </div>

        <style>{`
          @keyframes pulse {
            0%,100% { opacity:1; }
            50% { opacity:0.3; }
          }
        `}</style>
      </div>
    );
  }

  // ── Main dashboard ─────────────────────────────────────────────────────────
  return (
    <div style={{ fontFamily: "Arial, sans-serif", background: "#f5f5f5", minHeight: "100vh" }}>

      {/* Header */}
      <div style={{
        background: "#0a0a1a",
        color: "white",
        padding: "16px 40px",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
      }}>
        <div>
          <h1 style={{ margin: 0, fontSize: "24px" }}>❤️ CARDIA — Cardiac Digital Twin</h1>
          <p style={{ margin: "4px 0 0", color: "#aaa", fontSize: "13px" }}>
            Real-time cardiac monitoring + AI coaching
          </p>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <div style={{
            width: "10px", height: "10px",
            borderRadius: "50%",
            background: "#00c853",
            boxShadow: "0 0 8px #00c853",
          }} />
          <span style={{ color: "#00c853", fontSize: "13px" }}>
            ESP32 Live • {lastUpdate}
          </span>
        </div>
      </div>

      {/* Safety Banner */}
      <div style={{
        background: safetyColor,
        color: "white",
        padding: "12px",
        textAlign: "center",
        fontSize: "18px",
        fontWeight: "bold",
        transition: "background 0.5s",
      }}>
        {data.safety === "DANGER"  && "🔴 DANGER — Stop Activity Immediately!"}
        {data.safety === "CAUTION" && "🟡 CAUTION — Reduce Intensity"}
        {data.safety === "SAFE"    && "🟢 SAFE — All Good!"}
      </div>

      <div style={{ padding: "25px 40px" }}>

        {/* Vitals Grid */}
        <div style={{
          display: "grid",
          gridTemplateColumns: "repeat(4,1fr)",
          gap: "16px",
          marginBottom: "20px",
        }}>
          <StatCard title="Heart Rate"    value={data.heart_rate} unit="BPM"  color="#e53935" />
          <StatCard title="HRV RMSSD"    value={data.hrv}        unit="ms"   color="#1565c0" />
          <StatCard title="SpO2"          value={data.spo2}       unit="%"    color="#2e7d32" />
          <StatCard title="Activity Load" value={data.load}       unit="/10"  color="#6a1b9a" />
        </div>

        {/* CES + Zone */}
        <div style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: "16px",
          marginBottom: "20px",
        }}>
          {/* CES */}
          <div style={card}>
            <h3 style={cardTitle}>Cardiac Enhancement Score</h3>
            <div style={{ fontSize: "60px", fontWeight: "bold", color: "#1565c0", textAlign: "center" }}>
              {data.ces}
            </div>
            <div style={{ textAlign: "center", color: "#888", fontSize: "13px" }}>out of 100</div>
            <div style={{ background: "#eee", borderRadius: "8px", height: "8px", marginTop: "12px" }}>
              <div style={{
                background: data.ces > 70 ? "#00c853" : data.ces > 40 ? "#ff9100" : "#ff1744",
                width: `${data.ces}%`,
                height: "8px",
                borderRadius: "8px",
                transition: "width 0.8s ease",
              }} />
            </div>
          </div>

          {/* Zone */}
          <div style={card}>
            <h3 style={cardTitle}>HR Zone</h3>
            <div style={{
              fontSize: "44px",
              fontWeight: "bold",
              color: zoneColors[data.zone],
              textAlign: "center",
              transition: "color 0.5s",
            }}>
              {zoneNames[data.zone]}
            </div>
            <div style={{ display: "flex", gap: "5px", marginTop: "18px" }}>
              {zoneColors.map((c, i) => (
                <div key={i} style={{
                  flex: 1, height: "10px", borderRadius: "4px",
                  background: c,
                  opacity: i === data.zone ? 1 : 0.25,
                  transition: "opacity 0.5s",
                }} />
              ))}
            </div>
          </div>
        </div>

        {/* Heart Rate Graph */}
        <div style={{ ...card, marginBottom: "20px" }}>
          <h3 style={cardTitle}>Live Heart Rate — ESP32 + MAX30102</h3>
          <HeartGraph currentHR={data.heart_rate} />
        </div>

        {/* Strategy + Coach */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>

          {/* Strategy */}
          <div style={card}>
            <h3 style={cardTitle}>🎯 Recommended Strategy</h3>
            <div style={{
              background: "#e8f5e9",
              borderRadius: "10px",
              padding: "20px",
              textAlign: "center",
              fontSize: "20px",
              fontWeight: "bold",
              color: "#2e7d32",
              marginTop: "10px",
            }}>
              {data.strategy}
            </div>
            <div style={{
              marginTop: "10px",
              padding: "10px",
              background: "#f5f5f5",
              borderRadius: "8px",
              fontSize: "13px",
              color: "#666",
              textAlign: "center",
            }}>
              Source: {data.source === "esp32" ? "📡 ESP32 Sensor" : "💻 Simulation"}
            </div>
          </div>

          {/* Coach */}
          <div style={card}>
            <h3 style={cardTitle}>🤖 AI Cardiac Coach</h3>
            <div style={{
              background: "#f5f5f5",
              borderRadius: "10px",
              padding: "15px",
              fontSize: "14px",
              lineHeight: "1.7",
              color: "#333",
              minHeight: "90px",
              marginTop: "10px",
            }}>
              {data.coach}
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}

// ── Shared styles ─────────────────────────────────────────────────────────────
const card = {
  background: "white",
  borderRadius: "12px",
  padding: "20px",
  boxShadow: "0 2px 10px rgba(0,0,0,0.06)",
};

const cardTitle = {
  margin: "0 0 5px",
  fontSize: "14px",
  color: "#888",
  fontWeight: "600",
  textTransform: "uppercase",
  letterSpacing: "0.5px",
};

function StatCard({ title, value, unit, color }) {
  return (
    <div style={{
      ...card,
      textAlign: "center",
      borderTop: `4px solid ${color}`,
    }}>
      <div style={{ color: "#999", fontSize: "12px", marginBottom: "8px",
                    textTransform: "uppercase", letterSpacing: "0.5px" }}>
        {title}
      </div>
      <div style={{
        fontSize: "40px",
        fontWeight: "bold",
        color,
        transition: "all 0.5s",
      }}>
        {value}
      </div>
      <div style={{ color: "#bbb", fontSize: "12px", marginTop: "4px" }}>{unit}</div>
    </div>
  );
}

export default Dashboard;