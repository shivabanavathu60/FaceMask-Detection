document.addEventListener("DOMContentLoaded", function () {
    const startButton = document.getElementById("start");
    const stopButton = document.getElementById("stop");
    const videoFeed = document.getElementById("video_feed");

    function toggleDetection(status) {
        fetch("/toggle_detection", { method: "POST" })
            .then(response => response.json())
            .then(data => {
                console.log("Detection Status:", data.status);
                if (data.status === "started") {
                    videoFeed.src = "/video_feed";  // Start video feed
                } else {
                    videoFeed.src = "";  // Stop video feed
                }
            })
            .catch(error => console.error("Error:", error));
    }

    startButton.addEventListener("click", () => toggleDetection(true));
    stopButton.addEventListener("click", () => toggleDetection(false));
});
