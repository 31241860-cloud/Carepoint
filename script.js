document.querySelectorAll("[data-example]").forEach((button) => {
  button.addEventListener("click", () => {
    const box = document.querySelector("#problem");
    if (box) {
      box.value = button.dataset.example;
      box.focus();
    }
  });
});