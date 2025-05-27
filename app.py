import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { useState } from "react";
import { Bar, Line } from "react-chartjs-2";
import { motion } from "framer-motion";

export default function FacebookDashboard() {
  const [accessToken, setAccessToken] = useState("");
  const [postId, setPostId] = useState("");
  const [pageId, setPageId] = useState("");
  const [comments, setComments] = useState([]);
  const [sentimentData, setSentimentData] = useState(null);
  const [engagementData, setEngagementData] = useState(null);

  const fetchData = async () => {
    // Simulated sentiment data
    setSentimentData({
      labels: ["Positive", "Neutral", "Negative"],
      datasets: [
        {
          label: "Sentiment Count",
          data: [12, 5, 3],
        },
      ],
    });
    // Simulated engagement data
    setEngagementData({
      labels: ["Post 1", "Post 2", "Post 3"],
      datasets: [
        {
          label: "Likes",
          data: [100, 80, 120],
        },
        {
          label: "Comments",
          data: [20, 30, 25],
        },
      ],
    });
    setComments([
      { text: "Love this!", sentiment: "Positive" },
      { text: "Okay, I guess.", sentiment: "Neutral" },
      { text: "This is bad.", sentiment: "Negative" },
    ]);
  };

  return (
    <motion.div
      className="grid gap-4 p-4 md:grid-cols-2 xl:grid-cols-3"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <Card className="col-span-full">
        <CardContent className="flex flex-col gap-4 md:flex-row md:items-end">
          <Input
            placeholder="Access Token"
            value={accessToken}
            onChange={(e) => setAccessToken(e.target.value)}
          />
          <Input
            placeholder="Facebook Post ID"
            value={postId}
            onChange={(e) => setPostId(e.target.value)}
          />
          <Input
            placeholder="Page ID (for Insights)"
            value={pageId}
            onChange={(e) => setPageId(e.target.value)}
          />
          <Button onClick={fetchData}>Analyze</Button>
        </CardContent>
      </Card>

      {sentimentData && (
        <Card>
          <CardContent>
            <h2 className="text-xl font-bold mb-2">Sentiment Distribution</h2>
            <Bar data={sentimentData} />
          </CardContent>
        </Card>
      )}

      {engagementData && (
        <Card className="col-span-2">
          <CardContent>
            <h2 className="text-xl font-bold mb-2">Engagement Overview</h2>
            <Line data={engagementData} />
          </CardContent>
        </Card>
      )}

      <Card className="col-span-full">
        <CardContent>
          <h2 className="text-xl font-bold mb-2">Comments</h2>
          <ul className="space-y-2">
            {comments.map((c, i) => (
              <li key={i} className="p-2 rounded bg-gray-100">
                <strong>{c.sentiment}:</strong> {c.text}
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>
    </motion.div>
  );
}
