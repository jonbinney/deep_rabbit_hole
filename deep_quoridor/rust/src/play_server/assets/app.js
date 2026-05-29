// Replaced in Task 8.
fetch("/api/config")
  .then((r) => r.json())
  .then((cfg) => {
    document.getElementById("status").textContent =
      "Server up. Board " + cfg.board_size + "x" + cfg.board_size +
      ", " + cfg.models.length + " model(s) available.";
  });
