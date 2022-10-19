import React from 'react';

import Button from 'react-bootstrap/Button';

export default function DownloadButton({buttonText, handleClick}) {
    return (
        <div className="d-grid gap-2 mt-5 mb-5">
            <Button onClick={handleClick}>{buttonText}</Button>
        </div>
    )
}
