document.addEventListener("DOMContentLoaded", function () {
    const startButton = document.getElementById("start");
    const stopButton = document.getElementById("stop");
    const videoFeed = document.getElementById("video_feed");
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    let stream = null;
    let interval = null;

    async function startWebcam() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
            videoFeed.srcObject = stream;

            // Capture frames and send to Flask backend for processing
            interval = setInterval(() => {
                ctx.drawImage(videoFeed, 0, 0, canvas.width, canvas.height);
                canvas.toBlob(blob => {
                    const formData = new FormData();
                    formData.append("frame", blob);
                    fetch("/process_frame", { method: "POST", body: formData })
                        .then(response => response.json())
                        .then(data => console.log("Mask Detection:", data))
                        .catch(error => console.error("Error:", error));
                }, "image/jpeg");
            }, 1000); // Send frame every second
        } catch (error) {
            console.error("Camera access denied:", error);
        }
    }

    function stopWebcam() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
            clearInterval(interval);
        }
    }

    startButton.addEventListener("click", startWebcam);
    stopButton.addEventListener("click", stopWebcam);
});
