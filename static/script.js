// Random Emoji Muncul
function createEmoji() {
    const emojis = ["🐶", "🐱", "🐾", "💖", "🐕", "🐈", "🐩", "🐕‍🦺", "😸", "😹", "😻", "😼", "😽", "🙀", "😿", "🐱", "😾", "🐈‍⬛"];
    let emoji = document.createElement("div");
    emoji.innerHTML = emojis[Math.floor(Math.random() * emojis.length)];
    emoji.classList.add("emoji");
    
    // Posisi random di layar
    emoji.style.position = "absolute";
    emoji.style.left = Math.random() * window.innerWidth + "px";
    emoji.style.top = Math.random() * window.innerHeight + "px";
    emoji.style.fontSize = "24px";  // Ukuran emoji
    emoji.style.transition = "opacity 0.5s ease-out";  

    document.body.appendChild(emoji);

    // Hilangkan setelah 3 detik
    setTimeout(() => {
        emoji.style.opacity = "0";
        setTimeout(() => emoji.remove(), 500); // Hapus setelah animasi
    }, 3000);
}

// Panggil fungsi setiap 1.5 detik
setInterval(createEmoji, 1500);
