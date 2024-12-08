function openSidebar() {
    document.getElementById("sidebar").style.width = "250px";
}

function closeSidebar() {
    document.getElementById("sidebar").style.width = "0";
}
// Collapsible menu functionality
document.addEventListener('DOMContentLoaded', function () {
    const collapsibles = document.querySelectorAll('.collapsible');

    collapsibles.forEach(button => {
        button.addEventListener('click', function () {
            this.classList.toggle('active');
            const content = this.nextElementSibling;
            if (content.style.display === 'block') {
                content.style.display = 'none';
            } else {
                content.style.display = 'block';
            }
        });
    });

    // PDF rendering functionality
    const url = "/static/files/AI-EMG-FilteringSpasticity-TFG-JMF.pdf"; // Path to your PDF
    const pdfjsLib = window["pdfjs-dist/build/pdf"];
    pdfjsLib.GlobalWorkerOptions.workerSrc = "/static/js/pdf.worker.js";

    let pdfDoc = null,
        currentPage = 1,
        totalPages = 0,
        scale = 1,
        canvas = document.getElementById("pdf-render"),
        ctx = canvas.getContext("2d");

    const renderPage = (num) => {
        pdfDoc.getPage(num).then((page) => {
            const viewport = page.getViewport({ scale });
            canvas.height = viewport.height;
            canvas.width = viewport.width;

            const renderContext = {
                canvasContext: ctx,
                viewport: viewport,
            };
            page.render(renderContext);

            document.getElementById("current-page").textContent = num;
        });
    };

    const queueRenderPage = (num) => {
        renderPage(num);
    };

    document.getElementById("prev-page").addEventListener("click", () => {
        if (currentPage <= 1) return;
        currentPage--;
        queueRenderPage(currentPage);
    });

    document.getElementById("next-page").addEventListener("click", () => {
        if (currentPage >= totalPages) return;
        currentPage++;
        queueRenderPage(currentPage);
    });

    pdfjsLib.getDocument(url).promise.then((pdfDoc_) => {
        pdfDoc = pdfDoc_;
        totalPages = pdfDoc.numPages;
        document.getElementById("total-pages").textContent = totalPages;

        renderPage(currentPage);
    });
});
document.getElementById('file').addEventListener('change', function() {
    const file = this.files[0];
    if (file && !file.name.endsWith('.csv')) {
        alert('Invalid file type. Please upload a CSV file.');
        this.value = ''; // Clear the file input
    }
});

document.querySelector('form').addEventListener('submit', async function(event) {
    event.preventDefault();
    const formData = new FormData(this);
    try {
        const response = await fetch(this.action, {
            method: 'POST',
            body: formData,
        });
        const result = await response.json();
        if (!response.ok) {
            alert(result.error || 'An error occurred.');
        } else {
            window.location.href = response.url; // Redirect on success
        }
    } catch (error) {
        alert('An unexpected error occurred.');
    }
});
