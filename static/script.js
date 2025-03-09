document.addEventListener("DOMContentLoaded", function () {
    const startButton = document.getElementById("start");
    const stopButton = document.getElementById("stop");
    const videoFeed = document.getElementById("video_feed");

    function toggleDetection(start) {
        fetch("/toggle_detection", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ status: start ? "start" : "stop" }) // Explicitly send start/stop status
        })
        .then(response => response.json())
        .then(data => {
            console.log("Detection Status:", data.status);
            if (data.status === "started") {
                videoFeed.src = "/video_feed";  // Start video feed
                videoFeed.style.display = "block";
            } else {
                videoFeed.src = "";  // Stop video feed
                videoFeed.style.display = "none";
            }
        })
        .catch(error => console.error("Error:", error));
    }

    startButton.addEventListener("click", () => toggleDetection(true));
    stopButton.addEventListener("click", () => toggleDetection(false));
});
