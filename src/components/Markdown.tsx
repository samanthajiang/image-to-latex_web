import '@uiw/react-md-editor/markdown-editor.css';
import '@uiw/react-markdown-preview/markdown.css';
import { useState } from 'react';

import dynamic from 'next/dynamic';

const MDEditor = dynamic(() => import('@uiw/react-md-editor'), { ssr: false });

function Markdown() {
  const [value, setValue] = useState('**Hello world!!!**');
  return (
    <div>
      <MDEditor value={value} onChange={setValue} hideToolbar='True'/>
    </div>
  );
}

export default Markdown;
