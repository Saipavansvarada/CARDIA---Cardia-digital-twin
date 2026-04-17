import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { useState, useEffect } from "react";

function HeartGraph({ currentHR }) {
  const [data, setData] = useState([]);

  useEffect(() => {
    if (currentHR && currentHR !== "--") {
      setData((prev) => {
        const newPoint = { time: prev.length, hr: parseFloat(currentHR) };
        const updated  = [...prev, newPoint];
        return updated.slice(-30); // keep last 30 points
      });
    }
  }, [currentHR]);

  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
        <XAxis dataKey="time" hide />
        <YAxis domain={[40, 200]} />
        <Tooltip
          formatter={(val) => [`${val} BPM`, "Heart Rate"]}
          labelFormatter={() => ""}
        />
        <Line
          type="monotone"
          dataKey="hr"
          stroke="#e53935"
          strokeWidth={2}
          dot={false}
          animationDuration={300}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

export default HeartGraph;