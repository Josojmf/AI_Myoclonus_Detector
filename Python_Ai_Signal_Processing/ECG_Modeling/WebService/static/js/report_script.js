document.addEventListener('DOMContentLoaded', function () {
    const pdfContainer = document.getElementById('pdf-container');
    const canvas = document.getElementById('pdf-render');
    const prevPageButton = document.getElementById('prev-page');
    const nextPageButton = document.getElementById('next-page');
    const currentPageElement = document.getElementById('current-page');
    const totalPagesElement = document.getElementById('total-pages');

    const context = canvas.getContext('2d');
    let pdfDoc = null;
    let pageNum = 1;
    let pageRendering = false;
    let pageNumPending = null;

    const scale = 1.5; // Adjust this to scale the PDF

    // Path to your PDF file
    const url = 'http://127.0.0.1:5000/static/AI-EMG-FilteringSpasticity-TFG-JMF.pdf';

    // Load the PDF.js library
    const pdfjsLib = window['pdfjs-dist/build/pdf'];
    pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.14.305/pdf.worker.min.js';

    console.log("PDF.js initialized");

    // Render the page
    function renderPage(num) {
        console.log("Rendering page:", num);
        pageRendering = true;

        // Get the page
        pdfDoc.getPage(num).then((page) => {
            const viewport = page.getViewport({ scale });

            console.log("Viewport dimensions:", viewport.width, viewport.height);

            canvas.height = viewport.height;
            canvas.width = viewport.width;

            const renderContext = {
                canvasContext: context,
                viewport: viewport,
            };
            const renderTask = page.render(renderContext);

            // Wait for rendering to finish
            renderTask.promise.then(() => {
                console.log("Page rendered successfully");
                pageRendering = false;

                if (pageNumPending !== null) {
                    renderPage(pageNumPending);
                    pageNumPending = null;
                }
            }).catch((err) => {
                console.error("Error rendering page:", err);
            });
        }).catch((err) => {
            console.error("Error fetching page:", err);
        });

        // Update the page info
        currentPageElement.textContent = num;
    }

    // Load the PDF
    pdfjsLib.getDocument(url).promise.then((pdf) => {
        console.log("PDF loaded successfully:", pdf);
        pdfDoc = pdf;
        totalPagesElement.textContent = pdf.numPages;
        renderPage(pageNum);
    }).catch((error) => {
        console.error("Error loading PDF:", error);
    });

    // Queue rendering for another page
    function queueRenderPage(num) {
        if (pageRendering) {
            pageNumPending = num;
        } else {
            renderPage(num);
        }
    }

    // Navigate to the previous page
    prevPageButton.addEventListener('click', () => {
        if (pageNum <= 1) {
            return;
        }
        pageNum--;
        queueRenderPage(pageNum);
    });

    // Navigate to the next page
    nextPageButton.addEventListener('click', () => {
        if (pageNum >= pdfDoc.numPages) {
            return;
        }
        pageNum++;
        queueRenderPage(pageNum);
    });
});
