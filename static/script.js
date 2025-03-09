document.getElementById("start").addEventListener("click", function() {
    fetch("/toggle_detection", { method: "POST" })
    .then(response => response.json())
    .then(data => console.log(data));
});

document.getElementById("stop").addEventListener("click", function() {
    fetch("/toggle_detection", { method: "POST" })
    .then(response => response.json())
    .then(data => console.log(data));
});
