document.addEventListener('DOMContentLoaded', () => {
    const videoUploadInput = document.getElementById('video-upload');
    const uploadButton = document.getElementById('upload-button');
    const uploadStatus = document.getElementById('upload-status');
    const generateResultButton = document.getElementById('generate-result-button');
    const processingStatus = document.getElementById('processing-status');
    const processedVideoPlayer = document.getElementById('processed-video-player');

    // This variable will now primarily be used for the 'generate result' button
    // The initial state of 'uploadedFilename' should be derived from the URL if a video was just uploaded
    let uploadedFilename = null;

    // --- NEW: Check for video_filename in URL on page load ---
    const urlParams = new URLSearchParams(window.location.search);
    const initialVideoFilename = urlParams.get('video_filename');

    if (initialVideoFilename) {
        uploadedFilename = initialVideoFilename;
        uploadStatus.textContent = `File "${initialVideoFilename}" uploaded. Click "Generate Live Result" to process.`;
        uploadStatus.style.color = '#28a745'; // Green for success
        generateResultButton.style.display = 'block'; // Show the generate button
        // Optionally, hide other elements if they are visible
        processedVideoPlayer.style.display = 'none';
        processedVideoPlayer.src = '';
    } else {
        // If no filename in URL, ensure generate button is hidden initially
        generateResultButton.style.display = 'none';
        processedVideoPlayer.style.display = 'none';
    }
    // --- END NEW URL CHECK ---


    // Simulate clicking the hidden file input when the custom label is clicked
    document.querySelector('.upload-label').addEventListener('click', () => {
        videoUploadInput.click();
    });

    videoUploadInput.addEventListener('change', () => {
        if (videoUploadInput.files.length > 0) {
            uploadStatus.textContent = `File selected: ${videoUploadInput.files[0].name}`;
            uploadStatus.style.color = '#f8f9fa';
            
            // When a new file is selected, reset processing status and hide processed video
            processingStatus.textContent = ''; // Clear processing status
            generateResultButton.style.display = 'none'; // Hide generate button if new file selected
            processedVideoPlayer.style.display = 'none'; // Hide video player
            processedVideoPlayer.src = ''; // Clear video source
            uploadedFilename = null; // Clear the stored filename
        } else {
            uploadStatus.textContent = '';
            // If file selection is cleared, also reset things
            processingStatus.textContent = '';
            generateResultButton.style.display = 'none';
            processedVideoPlayer.style.display = 'none';
            processedVideoPlayer.src = '';
            uploadedFilename = null;
        }
    });

    uploadButton.addEventListener('click', async () => {
        const file = videoUploadInput.files[0];
        if (!file) {
            uploadStatus.textContent = 'Please select a video file first.';
            uploadStatus.style.color = '#dc3545'; // Red for error
            return;
        }

        const formData = new FormData();
        formData.append('video', file);

        uploadStatus.textContent = 'Uploading video...';
        uploadStatus.style.color = '#ffc107'; // Yellow for warning/in progress
        uploadButton.disabled = true; // Disable upload button during upload


        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                // --- THIS IS THE CRUCIAL CHANGE FOR REDIRECT AFTER UPLOAD ---
                // Instead of updating UI directly here, we redirect to the index page
                // with the filename as a query parameter. The index route will then
                // re-render the page with the 'Generate Live Result' button visible.
                window.location.href = `/?video_filename=${data.filename}`;
                // Note: The code below this line will not be executed after redirect.
                // It's left here for clarity of what used to happen.
                // uploadedFilename = data.filename;
                // uploadStatus.textContent = 'Upload successful! Click "Generate Live Result" to process.';
                // uploadStatus.style.color = '#28a745'; // Green for success
                // generateResultButton.style.display = 'block'; // Show the generate button
            } else {
                uploadStatus.textContent = `Upload failed: ${data.message}`;
                uploadStatus.style.color = '#dc3545';
                uploadButton.disabled = false; // Re-enable upload button on failure
            }
        } catch (error) {
            console.error('Error during upload:', error);
            uploadStatus.textContent = 'An error occurred during upload. Please try again.';
            uploadStatus.style.color = '#dc3545';
            uploadButton.disabled = false; // Re-enable upload button on error
        }
    });

    generateResultButton.addEventListener('click', async () => {
        // Now, uploadedFilename should be set either from a successful upload redirect
        // or from a direct visit to a URL with video_filename query parameter.
        if (!uploadedFilename) { // Check if the variable is populated
            processingStatus.textContent = 'No video selected for processing. Please upload one first.';
            processingStatus.style.color = '#dc3545';
            return;
        }

        processingStatus.textContent = 'Processing video... This may take a while.';
        processingStatus.style.color = '#ffc107'; // Yellow for warning/in progress
        generateResultButton.disabled = true; // Disable button during processing
        processedVideoPlayer.style.display = 'none'; // Hide player during re-processing

        try {
            const response = await fetch(`/process/${uploadedFilename}`, {
                method: 'POST'
            });

            const data = await response.json();

            if (data.success) {
                processingStatus.textContent = 'Video processed successfully!';
                processingStatus.style.color = '#28a745';
                processedVideoPlayer.src = data.processed_video_url;
                processedVideoPlayer.style.display = 'block'; // Show the video player
                // Assuming processed-video-player has a CSS transition for 'show-video'
                processedVideoPlayer.classList.add('show-video'); 
            } else {
                processingStatus.textContent = `Processing failed: ${data.message}`;
                processingStatus.style.color = '#dc3545';
            }
        } catch (error) {
            console.error('Error during processing:', error);
            processingStatus.textContent = 'An error occurred during processing. Please try again.';
            processingStatus.style.color = '#dc3545';
        } finally {
            generateResultButton.disabled = false; // Re-enable button
        }
    });
});